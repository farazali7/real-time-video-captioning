import torch
import torch.nn as nn
from generativeimage2text.layers.decoder import CaptioningModel, BeamHypotheses, top_k_top_p_filtering, \
    GeneratorWithBeamSearch
from generativeimage2text.model import get_image_encoder, TransformerDecoderTextualHead
from generativeimage2text.layers.decoder import convert2valid
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.torch_common import load_state_dict
from transformers import BertTokenizer
import functools
from torch.nn import functional as F
import timm
import lightning as L
import numpy as np

from src.utils.masking import create_padding_mask, create_casual_mask
import src.metrics as metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyVIT(nn.Module):
    def __init__(self, model_name: str):
        super(TinyVIT, self).__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=True,
                                       features_only=True)
        # TODO: Might need this
        # data_config = timm.data.resolve_model_data_config(self.model)
        # transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, x):
        out = self.model(x)

        return out


class StudentCandidateV1(nn.Module):
    """
    Student model V1 candidate for video captioning tasks. Uses TinyVIT + Transformer Decoder architectures.
    """

    def __init__(self, image_enc_name: str, d_model: int, n_head: int,
                 d_ffn: int, dropout: float, num_decoder_layers: int,
                 vocab_length: int, cls_token_id: int, sep_token_id: int):
        """ Constructor.

        Args:
            image_enc_name: Name of TinyVIT image encoder to load
            d_model: Hidden dimensionality of decoder layers
            n_head: Number of attention heads in decoder layers
            d_ffn: Hidden dimensionality of feed forward networks in decoder layers
            dropout: Dropout rate
            num_decoder_layers: Number of stacked decoder layers in decoder module
            vocab_length: Length of vocabulary for captions
            cls_token_id: Start token id
            sep_token_id: End token id
        """
        super(StudentCandidateV1, self).__init__()
        self.image_encoder = TinyVIT(image_enc_name)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head,
                                                        dim_feedforward=d_ffn, dropout=dropout,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        self.embed = nn.Embedding(vocab_length, d_model)
        self.linear = nn.Linear(d_model, vocab_length)
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        # Make feature map projectors
        self.projectors = nn.Sequential(*[nn.LazyLinear(1024) for _ in range(4)])

        self.pos_enc = PositionalEncoding(d_model=d_model)

    def forward(self, x, y):
        # x shape: [B, F, C, H, W]
        # Combine frames axis with batch to get shape: [BxF, C, H, W]
        init_shape = x.shape
        x = x.view(init_shape[0] * init_shape[1], *init_shape[2:])

        # Image encoder (TinyVIT) has 4 stages that give output here
        # Stage 1 is convolution based, Stages 2-4 are attention based
        image_enc_fmaps = self.image_encoder(x)

        # Take last feature map, average spatially, and restore frames as token length [B, F, De]
        memory = torch.mean(image_enc_fmaps[-1], dim=[2, 3]).view(init_shape[0], init_shape[1], -1)

        # Create padding and causal masks for captions
        pad_mask = create_padding_mask(y).to(device)
        tgt_mask = create_casual_mask(y.shape[1]).to(device)
        tgt_embed = self.embed(y)
        tgt_embed = self.pos_enc(tgt_embed)
        tgt_embed = tgt_embed / torch.sqrt(torch.tensor(self.embed.embedding_dim))
        out = self.decoder(tgt=tgt_embed, memory=memory,
                           tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask, tgt_is_causal=True)
        out = self.linear(out)

        return image_enc_fmaps + [out]

    def greedy_decode(self, src: torch.Tensor, max_len: int = 20):
        self.decoder.eval()
        batch_size = src.size(0)
        # tgt Shape: [B, 1]
        tgt = torch.tensor([self.cls_token_id]*batch_size, dtype=torch.long).unsqueeze(1).to(device)
        for i in range(max_len):
            with torch.no_grad():
                output = self.forward(src, tgt)[-1]
            output = torch.argmax(output, dim=-1)  # Assuming this gives you [batch_size, current_seq_length]
            last_tokens = output[:, -1].unsqueeze(-1)  # Correctly select the last token for each item in the batch
            tgt = torch.cat((tgt, last_tokens), dim=1)  # Concatenate along the sequence length dimension
            if torch.all(last_tokens.squeeze(-1) == self.sep_token_id):
                break

        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=500):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + pos  # Add the position encoding to original vector x
        return x


class GenerativeImageTextModel(CaptioningModel):
    """
    Generative Image 2 Text (GIT) Model reimplementation with custom behaviour changes to help with
    knowledge distillation.
    """

    def __init__(self, image_encoder, text_decoder, decoder, tokenizer, param):
        """ GIT constructor.

        Args:
            image_encoder: Model used for encoding input images
            text_decoder: Model used for textual logits as output
            decoder: Model used for decoding into text
            tokenizer: Model used for tokenizing input text
            param: Additional configurations
        """
        super().__init__(
            image_encoder,
            text_decoder,
            decoder=decoder,
            sos_index=tokenizer.cls_token_id,
            eos_index=tokenizer.sep_token_id,
            tokenizer=tokenizer,
            use_history_for_infer=True,
            loss_type='smooth',
            num_image_with_embedding=param.get('num_image_with_embedding')
        )

    @torch.no_grad()
    def forward_one_custom(self, batch, return_info=False):
        # This is a slight alteration to original method for getting output logits from the teacher model
        # shape: (batch_size, max_caption_length, vocab_size)
        if 'image' in batch:
            if isinstance(batch['image'], (list, tuple)):
                features = [self.image_encoder(im) for im in batch['image']]
                if self.num_image_with_embedding:
                    features = [f + e for f, e in zip(features, self.img_temperal_embedding)]
                if self.pooling_images is None:
                    visual_features = torch.cat(features, dim=1)
                elif self.pooling_images == 'avg':
                    visual_features = torch.stack(features, dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
            else:
                visual_features = self.image_encoder(batch['image'])
        else:
            visual_features = None
        visual_features_valid = None
        if 'context' in batch:
            context_embedding = self.context_embedding if self.context_not_share_embedding else self.textual.embedding
            all_context = [visual_features]
            all_valid = [convert2valid(visual_features.shape[:2])]
            for info in batch['context']:
                context = context_embedding(info['tokens'])
                valid = convert2valid(info['tokens'].shape, info['length'])
                all_context.append(context)
                all_valid.append(valid)
            visual_features = torch.cat(all_context, dim=1)
            visual_features_valid = torch.cat(all_valid, dim=1)

        # Get output logits
        has_image = (visual_features is not None)
        assert has_image == ('image' in batch)
        #if self.use_masked_as_input_for_train:
            #caption_token_input = batch["masked_caption_tokens"]
        #else:
        caption_token_input = batch["caption_tokens"]
        #caption_lengths = batch["caption_lengths"]

        output_logits = self.textual(
            visual_features,
            caption_token_input,
            #caption_lengths=caption_lengths,
            hidden_valid_mask=visual_features_valid,
            bi_valid_mask_caption=batch.get('bi_valid_mask_caption'),
        )

        return output_logits

    def infer(self, batch, visual_features, visual_features_valid, search_param=None):
        batch_size = visual_features.size(0)
        if 'prefix' not in batch:
            start_predictions = visual_features.new_full(
                (batch_size, 1), self.sos_index
            ).long()
        else:
            # if batch size is larger than 1, the prefix length could be
            # different, and we have to padding non-valid data, which
            # is not supported
            assert len(batch['prefix']) == 1, 'not supported'
            start_predictions = batch['prefix'].long()

        self.prev_encoded_layers = None
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        decoding_step = functools.partial(
            self.decoding_step, visual_features, visual_features_valid,
            batch.get('bi_valid_mask_caption')
        )

        search_param = search_param or {}
        # the start_predictions are not in predicted_caption
        predicted_caption, logprobs, logits_dict = self.decoder.search(
            start_predictions, decoding_step, **search_param
        )

        if 'prefix' in batch:
            # we need to remove prefix from predicted_caption
            predicted_caption = predicted_caption[:, start_predictions.shape[1]:]
        output_dict = {
            'predictions': predicted_caption,
            'logprobs': logprobs,
            'logits_dict': logits_dict,
            'visual_features': visual_features,
        }
        return output_dict


class GeneratorWithBeamSearchV2(GeneratorWithBeamSearch):
    """
    Decoder model used in GIT but with custom search functionality.
    """

    def __init__(self, *args, **kwargs):
        """ Constructor.

        Args:
            *args: Initialization arguments
            **kwargs: Initialization keyword arguments
        """
        super(GeneratorWithBeamSearchV2, self).__init__(*args, **kwargs)

    def search(self, input_ids, step, num_keep_best=1, do_sample=False, top_k=None, top_p=None,
               num_return_sequences=1):
        if num_return_sequences != 1:
            input_ids = input_ids[:, None, :].expand(
                input_ids.shape[0], num_return_sequences, input_ids.shape[1])
            input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        batch_size, cur_len = input_ids.shape
        num_beams = self.beam_size
        pad_token_id = self._eos_index
        eos_token_ids = [self._eos_index]
        per_node_beam_size = self.per_node_beam_size
        repetition_penalty = self.repetition_penalty
        temperature = self.temperature

        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams,
                                                cur_len)  # (batch_size * num_beams, cur_len)

        # prefix_len = cur_len
        # max_length = self.max_steps + prefix_len
        max_length = self.max_steps
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, self.length_penalty, early_stopping=False) for _ in
            range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        ## cache compute states
        # past = None

        # done sentences
        done = [False for _ in range(batch_size)]
        saved_logits = []
        while cur_len < max_length:
            scores = step(input_ids)  # (batch_size * num_beams, cur_len, vocab_size)
            vocab_size = scores.shape[-1]
            saved_logits.append([i.detach().cpu().numpy() for i in scores])

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty
            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p,
                                               min_tokens_to_keep=2)  # (batch_size * num_beams, vocab_size)
                # Sample [per_node_beam_size] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                                               num_samples=per_node_beam_size)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, per_node_beam_size)
                next_scores = _scores + beam_scores[:, None].expand_as(
                    _scores)  # (batch_size * num_beams, per_node_beam_size)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams, device=next_words.device) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, per_node_beam_size)
                next_words = next_words.view(batch_size,
                                             per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size,
                                               per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                original_scores = scores
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, per_node_beam_size * num_beams, dim=1, largest=True,
                                                     sorted=True)
            assert next_scores.size() == next_words.size() == (batch_size, per_node_beam_size * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(),
                                                     score.item())
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            # if past:
            # reordered_past = []
            # for layer_past in past:
            ## get the correct batch idx from layer past batch dim
            ## batch dim of `past` and `mems` is at 1st position
            # reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
            # reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
            ## check that shape matches
            # assert reordered_layer_past.shape == layer_past.shape
            # reordered_past.append(reordered_layer_past)
            # past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                               dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores, min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        if num_keep_best == 1:
            decoded = decoded.squeeze(dim=1)
        return decoded, logprobs, saved_logits


def get_git_model(tokenizer, param):
    image_encoder = get_image_encoder(
        param.get('image_encoder_type', 'CLIPViT_B_16'),
        input_resolution=param.get('test_crop_size', 224),
    )

    text_decoder = TransformerDecoderTextualHead(
        visual_feature_size=param.get('visual_feature_size', 768),
        vocab_size=30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768 * 4,
        max_caption_length=1024,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )

    decoder = GeneratorWithBeamSearchV2(
        eos_index=tokenizer.sep_token_id,
        max_steps=15,
        # max_steps=1024,
        beam_size=4,
        length_penalty=0.6,
    )

    model = GenerativeImageTextModel(
        image_encoder,
        text_decoder,
        decoder=decoder,
        tokenizer=tokenizer,
        param=param
    )

    return model


class GenerativeImageTextTeacher(nn.Module):
    """
    Generative Image 2 Text Teacher Model (Wrapper with tokenizer and params for instantiation)
    """

    def __init__(self, param_path: str, pretrained_weights: str):
        """ Constructor.

        Args:
            param_path: Path to model configuration YAML
        """
        super(GenerativeImageTextTeacher, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.param = load_from_yaml_file(param_path)
        self.model = get_git_model(self.tokenizer, self.param)
        ckpt = torch.load(pretrained_weights)
        ckpt = ckpt['model']
        load_state_dict(self.model, ckpt)

        # Freeze model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Deactivate behaviour of BN and Dropout layers
        self.model.eval()

    def forward_output_logits(self, x, y):
        list_of_sequences = [list(i) for i in x]
        out = []
        for i, seq in enumerate(list_of_sequences):
            imgs = [i.unsqueeze(0) for i in seq]
            batch = {'image': imgs,
                     'caption_tokens': y[i].unsqueeze(0)}
            res = self.model.forward_one_custom(batch=batch)
            out.append(res)
        return out

    def forward(self, x):
        list_of_sequences = [list(i) for i in x]
        out = []
        for seq in list_of_sequences:
            imgs = [i.unsqueeze(0) for i in seq]
            with torch.no_grad():
                result = self.model({
                    'image': imgs
                })
                cap = self.tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
                n = min(len(cap.split(' ')), len(result['logits_dict']))

                # Get the distribution of logits for each beam for each of the predicted n words
                # Shape: [n, 4, v] where n = number of words, 4 = number of beams, v = vocab length
                beam_logits_distributions = torch.from_numpy(np.array(result['logits_dict'][:n]))

                # Get the actual predicted word tokens (indices) and change shape to have same no. of dimensions
                # as the logit distribution tensors, Shape: [n, 4, 1]
                word_tokens = result['predictions'][0, 1:n+1]
                word_tokens = word_tokens[:, None, None].expand(-1, 4, -1)

                # Get indices of the beam in each word that has the highest logit at the
                indices = torch.gather(beam_logits_distributions, dim=2, index=word_tokens).squeeze().argmax(dim=1)

                indices_expanded = indices[:, None, None].expand(-1, -1, beam_logits_distributions.shape[-1])
                res = torch.gather(beam_logits_distributions, dim=1, index=indices_expanded).squeeze()[None, ...]

                result['output'] = res
                result['cap'] = cap
                out.append(result)

        return out


class DistillationTrainer(L.LightningModule):
    """
    PyTorch Lightning module for knowledge distillation training
    """

    def __init__(self, teacher, student, lr):
        """ Constructor.

        Args:
            teacher: Teacher model
            student: Student model
            lr: Learning rate
        """
        super(DistillationTrainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.fmap_distill_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore the padding from the dataloader

        self.teacher_activations = {}
        # Create hooks for teacher feature/attention maps we want
        self.wanted_block_indices = torch.arange(0, 23, 6)
        for i, block_idx in enumerate(self.wanted_block_indices):
            self.teacher.model.image_encoder.transformer.resblocks[block_idx].register_forward_hook(self.get_teacher_activation(i))

    def training_step(self, batch, batch_idx):
        self.teacher.eval()

        x, y = batch['frames'], batch['caption']

        out_student = self.student(x, y)
        out_teacher = self.teacher.forward_output_logits(x, y)

        # LOSS 1: Get feature maps and match and compute loss
        # Student activations
        student_fmaps = [torch.mean(fmap, dim=[2, 3]) for fmap in out_student[:-1]]

        # Teacher activations (get cls token)
        teacher_fmaps = [torch.stack(fmap)[:, 0, ...].squeeze() for fmap in self.teacher_activations.values()]

        # Project student fmaps to teacher dimensionality
        student_fmaps = [self.student.projectors[i](fmap) for i, fmap in enumerate(student_fmaps)]

        # Compute feature map distillation loss
        fmap_loss = self.fmap_distill_loss(torch.stack(teacher_fmaps), torch.stack(student_fmaps))

        # LOSS 2: Compute a loss between output logits of teacher and student
        # Student logits shape: [B, GT_length, vocab]
        # Teacher logits shape: [B, GT_length, vocab]
        temperature=2
        teacher_logits = torch.cat(out_teacher, dim=0)
        student_logits = out_student[-1]
        teacher_logits_kl=teacher_logits/temperature
        student_logits_kl=student_logits/temperature
        kl_loss = self.kl_div_loss(student_logits_kl.log_softmax(dim=-1), teacher_logits_kl.softmax(dim=-1))
        kl_loss=kl_loss*(temperature ** 2)

        # LOSS 3: Compute loss between output of student and GT
        y_target = y[:, 1:].reshape(-1)
        y_pred = student_logits[:, :-1].reshape(-1, student_logits.shape[2])

        ce_loss = self.ce_loss(y_pred, y_target)

        loss = ce_loss + kl_loss + fmap_loss

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_fmap_loss", fmap_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Clear the teacher activations for this batch
        del self.teacher_activations
        self.teacher_activations = {}

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['caption']

        out_student = self.student.greedy_decode(x, max_len=y.shape[-1]+5)
        preds = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in out_student]
        caps = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in y]

        # Add BLEU for student
        caps = [[c] for c in caps]
        loss = metrics.calculate_bleu_score_corpus(caps, preds)
        add_loss=metrics.calculate_meteor_score_corpus(caps, preds)
        rouge_loss=metrics.calculate_rouge_score(caps, preds)
        print(f'Ground-Truth Captions: {caps}')
        print(f'Student Predictions: {preds}')
        print(f'BLEU@4: {loss}')
        print(f'METEOR: {add_loss}')
        print(f'ROUGE: {rouge_loss}')

        self.log("val_loss", loss, prog_bar=True)



        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=4,
                                                                  min_lr=1e-8,
                                                                  factor=0.5)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor": "val_loss"}]

    def get_teacher_activation(self, name):
        def hook(model, input, output):
            if name in self.teacher_activations:
                self.teacher_activations[name].append(output.detach())
            else:
                self.teacher_activations[name] = [output.detach()]

        return hook

'''
def init(self, teacher, student, lr):
        """ Constructor.

        Args:
            teacher: Teacher model
            student: Student model
            lr: Learning rate
        """
        super(DistillationTrainer, self).init()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.inter_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce=nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['caption']
        self.teacher.eval()
        out_teacher = self.teacher(x)
        self.student.train()
        input_y=y[:,:-1]
        y_actual=y[:,1:].reshape(-1)
        out_student,student_visual = self.student(x,input_y)
        out_student_vec = out_student.view(-1, out_student.size(-1))
        y_actual = y_actual.view(-1)
        ce_loss= self.ce(out_student_vec,y_actual)

        #Need to do some adjustments to same shape to do intermediate of the final encoding
        #print(student_visual.shape)
        #visual_features_list = [item['visual_features'] for item in out_teacher]
        #visual_features_stacked = torch.stack(visual_features_list)
        #visual_features_stacked=visual_features_stacked.squeeze(1)
        #print(visual_features_stacked.shape)

        padding_vector = torch.full((1, 1, out_student.shape[-1]), float('-inf'))  # Filled with -inf
        padding_vector[:,:,self.tokenizer.pad_token_id] = 0  # Set the padding token index to 0 (or any desired value)
        #Need to do some adjustments to the final logits since student and teacher can output different number of tokens
        n=out_student.shape[1]
        n_teacher=0
        padded_tensors_list=[]
        for i in out_teacher:
            n_teacher=max(n_teacher,i['output'].shape[1])
        for i in out_teacher:
            if i['output'].shape[1]<n_teacher:
                padding = padding_vector.repeat(1, n_teacher-i['output'].shape[1], 1)
                padded_tensor = torch.cat([i['output'], padding], dim=1)
                padded_tensors_list.append(padded_tensor)
            else:
                padded_tensors_list.append(i['output'])

        combined_tensor = torch.cat(padded_tensors_list, dim=0)

        if n<n_teacher:
            padding = padding_vector.repeat(1, n_teacher-n, 1)
            out_student = torch.cat([out_student, padding], dim=1)
        else:
            padding = padding_vector.repeat(out_student.shape[0], n-n_teacher, 1)
            combined_tensor = torch.cat([combined_tensor, padding], dim=1)

        print(combined_tensor.shape)
        print(out_student.shape)

        kl_loss = self.kl_div_loss(out_student.log_softmax(dim=-1), combined_tensor.softmax(dim=-1))

        final_loss=kl_loss+ce_loss
        # Add knowledge distillation events here
        #loss = self.loss(out_student, out_teacher)

        self.log("train_loss", final_loss, prog_bar=True, on_step=False, on_epoch=True)

        return final_loss
'''