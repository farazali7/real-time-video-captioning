'''
Import Statements
'''
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
import torchvision.transforms.functional as TF
import hashlib
from torch.optim.lr_scheduler import OneCycleLR
import time
import av
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from src.utils.masking import create_padding_mask, create_casual_mask
import src.metrics as metrics
import os
from config import cfg
import uuid

'''
If you have NVIDIA CUDA
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
If you have Apple Metal Silicon
'''
# device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
        self.n_head=n_head
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head,
                                                        dim_feedforward=d_ffn, dropout=dropout,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        #We are calling an embedding function to embed our vocab tokens passed as input to decoder to d_model
        self.embed = nn.Embedding(vocab_length, d_model)
        #This will be to project it back to vocab
        self.linear = nn.Linear(d_model, vocab_length)
        #Default tokens from BERT tokenizer same used as teacher
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        # Make feature map projectors
        self.projectors = nn.Sequential(*[nn.LazyLinear(1024) for _ in range(4)])
        
        #These linear layers were created to project the final encoding dimension from encoder to match teacher visual features
        self.upsample = nn.LazyLinear(1542)
        self.project = nn.LazyLinear(1024)

        #This is for decoder distillation projection
        self.project_decoder=nn.LazyLinear(768)

        #This is to add positional encoding to the input target of decoder
        self.pos_enc = PositionalEncoding(d_model=d_model)
        #This was to see if adding encoder layers would help given the different images in a video can attend to eachother
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.temporal_encodings = nn.Parameter(torch.randn(6, d_model))


    def forward(self, x, y):
        # Get frame layer-wise feature maps and final visual representation from image encoder
        image_enc_fmaps, memory = self.forward_image_enc(x)

        # Pass labels and memory through decoder for final textual output
        out = self.forward_decoder(y, memory)

        return image_enc_fmaps + [out]

    def forward_image_enc(self, x):
        '''
        This could be useful method since we won't have to call encoder everytime in our greedy/beam decoders
        '''
        # x shape: [B, F, C, H, W]
        # Combine frames axis with batch to get shape: [BxF, C, H, W]
        # Check if the model is in training mode
        init_shape = x.shape
        x = x.view(init_shape[0] * init_shape[1], *init_shape[2:])

        # Image encoder (TinyVIT) has 4 stages that give output here
        # Stage 1 is convolution based, Stages 2-4 are attention based
        image_enc_fmaps = self.image_encoder(x)

        # Take last feature map, average spatially, and restore frames as token length [B, F, De]
        memory = torch.mean(image_enc_fmaps[-1], dim=[2, 3]).view(init_shape[0], init_shape[1], -1)
        #temporal_encodings_expanded = self.temporal_encodings.unsqueeze(0).expand_as(memory)
        #memory = memory + temporal_encodings_expanded
        #memory=self.transformer_encoder(memory)
        return image_enc_fmaps, memory

    def forward_decoder(self, y, memory):
        '''
        Again instead of having the forward call multiple times, this saves time
        '''
        #Add padding to ensure we aren't looking at parts of Y that have padding token
        pad_mask = create_padding_mask(y).to(device)
        #We embed our current y (could be caption if teacher forcing, or what we generated up until now)
        tgt_mask = create_casual_mask(y.shape[1]).to(device)
        #We then add positional encoding
        tgt_embed = self.embed(y)
        #As per the original paper we divide by Sqrt D our embeddings
        tgt_embed = self.pos_enc(tgt_embed)
        #We decode
        tgt_embed = tgt_embed / torch.sqrt(torch.tensor(self.embed.embedding_dim))
        #Project to vocab length
        out = self.decoder(tgt=tgt_embed, memory=memory,
                           tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask,tgt_is_causal=True)
        out = self.linear(out)
        #Output Result
        return out

    def greedy_decode(self, src: torch.Tensor, max_len: int = 10):
        '''
        Greedily decodes a brand new sequence
        Output: The brand new sequence
        '''
        #Ensure the decoder is in eval mode
        self.image_encoder.eval()
        with torch.no_grad():
            _, memory = self.forward_image_enc(src)

        self.decoder.eval()
        self.training = False
        batch_size = src.size(0)
        # tgt Shape: [B, 1]
        # Start off by creating a tensor of shape BxStart Token
        tgt = torch.tensor([self.cls_token_id]*batch_size, dtype=torch.long).unsqueeze(1).to(device)
        #While we haven't reached the max Length
        for i in range(max_len):
            #Run the forward prediction on our current sequence
            with torch.no_grad():
                output = self.forward_decoder(tgt, memory)
            #Get the best word predicted for each token
            output = torch.argmax(output, dim=-1) 
            #Grab the token from the last token predicted
            last_tokens = output[:, -1].unsqueeze(-1)
            #Add this token to our existing generated sequence, note we aren't embedding as this is taken care of in our forward
            tgt = torch.cat((tgt, last_tokens), dim=1)
            #If the token we just added was the end of sequence token we just end our sequence
            if torch.all(last_tokens.squeeze(-1) == self.sep_token_id):
                break

        return tgt
    
    def beam_search(self, src: torch.Tensor, max_len: int = 10, k: int = 3):
        '''
            We want to predict with beams our final sequence
        '''
        #Ensure the decoder and encoder is set to eval mode
        self.decoder.eval()
        self.image_encoder.eval()

        batch_size = src.size(0)
        # Initialize the start tokens and sequences Batch Size x 1
        tgt = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long).to(device)

        #We also know we will need to keep track of scores for each of these sequences Batch x K
        scores = torch.zeros(batch_size, k).to(device)
        # We ensure the sequences are expanded so its Batch Size x k
        sequences = tgt.unsqueeze(1).expand(-1, k, -1)
        #Create another tensor of batch sizex K^2 x 3 to help manage when we evaluate beams
        all_candidates = torch.empty(batch_size, k*k, 3, device=device)

        #Part of beam search to handle end of sequence
        '''# Finished sequences to add those beams that have completed with eos token generated
        finished_seqs = torch.full((batch_size, k, max_len), eos_token_id, dtype=torch.long, device=device)
        #The scores for each of these finished sequences
        finished_scores = torch.full((batch_size, k), 0, dtype=torch.float, device=device)
        #Keeping track of the finished counts for each batch to see if we have 5 sequences that have finished
        finished_counts = torch.zeros(batch_size, dtype=torch.long, device=device)'''
        
        #We first need to generate atleast one token for each beam
        with torch.no_grad():
            #We generate the memory for the images one time
            _, memory = self.forward_image_enc(src)
            #We get the output of the forward decoder
            decoder_output = self.forward_decoder(tgt, memory)
            #Get the log probabilities for all the tokens in the last predicted token
            log_probs = F.log_softmax(decoder_output[:, -1, :], dim=-1)
            #Use topk to find the best vocab indices and their respective scores
            scores, top_indices = log_probs.topk(k, dim=-1)
        #We add these top indices directly to our sequences, each beam represents one of the top 5 tokens predicted
        sequences = torch.cat([sequences, top_indices.unsqueeze(-1)], dim=-1)

        # Start beam search loop, since we predicted one token for each beam start at 2nd token
        for step in range(2, max_len):
            #For each beam we have to predict one, since each beam has different tgt now
            for i in range(k):
                #Grab all the sequences of the beams one at a time, but across the batch
                tgt = sequences[:, i]
                with torch.no_grad():
                    #Predict the next token
                    decoder_output = self.forward_decoder(tgt, memory)
                    #Get the log probabilities for the last token predicted
                    log_probs = F.log_softmax(decoder_output[:, -1, :], dim=-1)
                    #Grab the best scores and indices, this result would be Batch x K
                    top_scores, top_indices = log_probs.topk(k, dim=-1)  # BxK
                #Now we take the previous score for that beam and add the scores we just got.
                #Our scores for each batch would be size Bx1 where as our top_scores is Bxk so broadcasted to get Bxk scores for this beam
                local_scores = scores[:, i].unsqueeze(-1) + top_scores  # Bx1 + BxK
                #Now we figure out where in our all_candidates, the tensor to help in beam evaluation we should place
                offset = i * k
                all_candidates[:, offset:offset+k, 0] = local_scores
                all_candidates[:, offset:offset+k, 1] = i
                all_candidates[:, offset:offset+k, 2] = top_indices

            #Now that for this step we finished generating all the beams filled in all candidates lets evaluate by sorting.
            scores_to_sort = all_candidates[:,:,0].view(batch_size, -1)  # Shape: [batch_size, k*k]
            #We create create a sorted tensor
            sorted_scores, sorted_indices = scores_to_sort.sort(dim=1, descending=True)
            # Now, use sorted_indices to select top k candidates
            topk_indices = sorted_indices[:, :k]  # Get indices of top k scores for each batch
            # Initialize the new sequences tensor for this step, where this tensor should be one longer than our existing sequences since we are adding a new token
            new_sequences = torch.zeros(batch_size, k, step + 1, dtype=torch.long, device=device)
            # Extracting the beam indices and token indices from all_candidates using topk_indices
            for b in range(batch_size):
                #For each top candidate
                for idx in range(k):
                    # Get the global index from topk_indices, which points to the flat structure of all_candidates
                    global_idx = topk_indices[b, idx]

                    # Extract the beam index and token index for the current top candidate
                    beam_idx = all_candidates[b, global_idx, 1].long()  # Ensure it's used as an index
                    token_idx = all_candidates[b, global_idx, 2].long()

                    # Update the new_sequences tensor
                    new_sequences[b, idx, :-1] = sequences[b, beam_idx, :]  # Copy the previous sequence from selected beam
                    new_sequences[b, idx, -1] = token_idx  # Append the new token

                    # Update the score for this candidate
                    scores[b, idx] = all_candidates[b, global_idx, 0]  # Update score with the new score

                    #Part of Beam Search to handle end of sequence
                    '''
                    #If the token predicted was the EOS token and for this batch if we haven't predicted all the EOS beams yet
                        if token_idx == eos_token_id and finished_counts[b] < k:
                            # Store finished sequence
                            finished_seqs[b, finished_counts[b],0:step] = sequences[b, beam_idx]
                            finished_seqs[b, finished_counts[b], step] = token_idx  # Ensure EOS is included
                            finished_scores[b, finished_counts[b]] = new_score
                            finished_counts[b] += 1
                        else:
                            # Update the new_sequences tensor
                            new_sequences[b, idx, :-1] = sequences[b, beam_idx, :]  # Copy the previous sequence from selected beam
                            new_sequences[b, idx, -1] = token_idx  # Append the new token
                            
                            # Update the score for this candidate
                            scores[b, idx] = new_score
                    '''
            
            #Now we replace our sequences with new sequences
            sequences=new_sequences

        #Part of Beam Search to handle end of sequence
        '''# Select the best sequence for each batch
            final_sequences = []
            #We check to see for each batch
            for batch_idx in range(batch_size):
                #If there is any sequence that finished 
                if finished_counts[batch_idx] > 0:
                    #We take the one that had the largest score
                    best_finished_idx = finished_scores[batch_idx].argmax()
                    final_sequences.append(finished_seqs[batch_idx, best_finished_idx])
                #Otherwise we just take the beam that had the highest score at the end
                else:
                    best_seq_idx = scores[batch_idx].argmax()
                    final_sequences.append(sequences[batch_idx, best_seq_idx])

            return torch.stack(final_sequences)'''

        # Choose the sequence with the highest score
        final_sequences = sequences[torch.arange(batch_size), scores.argmax(dim=-1)]
        return final_sequences


class PositionalEncoding(nn.Module):
    '''
    Applies the vanilla positional encoding by the Transformer paper with Batch First
    '''
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
                #features = [self.image_encoder(im) for im in batch['image']]
                features = self.image_encoder(torch.stack(batch['image']).squeeze())
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
        hidden_states=[]
        for i in output_logits[1]:
            squeezed_i = i.squeeze(0)
            hidden_states.append(squeezed_i)
        hidden_states=torch.stack(hidden_states,dim=0)
        return output_logits[0],visual_features,hidden_states

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
        output_hidden_states=True,
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
        visual_features = []
        hidden_states = []
        for i, seq in enumerate(list_of_sequences):
            imgs = [i.unsqueeze(0) for i in seq]
            batch = {'image': imgs,
                     'caption_tokens': y[i].unsqueeze(0)}
            res, visual_feature, hidden_state = self.model.forward_one_custom(batch=batch)
            out.append(res)
            visual_features.append(visual_feature)
            hidden_states.append(hidden_state)
        return out, visual_features, hidden_states

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
                beam_logits_distributions = torch.from_numpy(np.array(result['logits_dict'][:n])).to(device)

                # Get the actual predicted word tokens (indices) and change shape to have same no. of dimensions
                # as the logit distribution tensors, Shape: [n, 4, 1]
                word_tokens = result['predictions'][0, 1:n+1].to(device)
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

    def __init__(self, teacher, student, lr, steps, epochs, logpath):
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

        # Loss 1
        self.fmap_distill_loss = nn.MSELoss()
        # Loss 4
        self.final_encoding_loss=nn.MSELoss()
        # Loss 2
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        # Loss 3
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        # Loss 5
        self.ce_loss2 = nn.CrossEntropyLoss()
        # Loss 6
        # self.decoder_distill_loss=nn.MSELoss()

        self.steps = steps
        self.epochs = epochs
        self.logpath = logpath

        # This is to store the teacher encoder activations
        self.teacher_encoder_activations = {}

        # This is to store the student decoder activations
        self.student_decoder_activations = {}

        # This is to store the teacher decoder activations
        self.teacher_decoder_activations = {}

        # Creating a directory to store the results of the run
        self.filename = f"_results_and_metrics.txt"

        # Create hooks to store the activations of the teacher encoder
        self.encoder_wanted_block_indices = torch.arange(0, 23, 6)
        self.teacher_encoder_hooks = []
        for i, block_idx in enumerate(self.encoder_wanted_block_indices):
            self.teacher_encoder_hooks.append(self.teacher.model.image_encoder.transformer.resblocks[block_idx].register_forward_hook(self.get_teacher_encoder_activation(i)))

        # Create hooks to store the activations of the student decoder
        self.student_decoder_hooks = []
        for layer_idx in range(self.student.decoder.num_layers):
            self.student_decoder_hooks.append(student.decoder.layers[layer_idx].register_forward_hook(self.get_student_decoder_activation(layer_idx)))

        # Create hooks to store the activations of the teacher decoder
        self.teacher_decoder_hooks = []
        for layer_idx in range(6):
            self.teacher_decoder_hooks.append(self.teacher.model.textual.transformer.encoder.layer[i].output.register_forward_hook(self.get_teacher_decoder_activation(layer_idx)))

        # To store the predictions for analysis
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Log configuration parameters to file
        with open(self.logpath + '/' + self.filename, 'a') as f:
            f.write(f'Results for the run: {self.filename}\n')
            f.write('\n************************************\n')
            f.write("\n" * 2)
            f.write(f'Teacher model: {teacher.__class__.__name__}\n')
            f.write(f"Teacher model configuration: {cfg['MODEL']['GenerativeImageTextTeacher']}\n")
            f.write('\n' * 2)
            f.write(f"Student model: {student.__class__.__name__}\n")
            f.write(f"Student model configuration: {cfg['MODEL']['StudentCandidateV1']}\n")
            f.write("\n" * 2)
            f.write("Parameters:\n")
            f.write(f"Learning Rate: {cfg['TRAIN']['LR']}\n")
            f.write(f"Number of epochs: {cfg['TRAIN']['TRAINER']['max_epochs']}\n")
            f.write(f"Batch size: {cfg['TRAIN']['BATCH_SIZE']}\n")
            f.write(f"Precision: {cfg['TRAIN']['TRAINER']['precision']}\n")
    
    def training_step(self, batch, batch_idx):
        # We ensure the teacher is in eval mode
        self.teacher.eval()

        # We grab from the dataloader the frames, caption and if needed caption-id to uniquely identify the caption used
        x, y, _, _ = batch['frames'], batch['caption'], batch['caption-id'], batch['vid-id']
        
        # We call the student forward function in two parts which outputs our
        # encoder feature maps, output logits, final encoder representation
        student_image_enc_fmaps, memory = self.student.forward_image_enc(x)

        # Output of the student decoder
        student_decoder_output = self.student.forward_decoder(y, memory)

        # The teacher outputs the teacher logits, and final encoder visual representations
        # and decoder hidden state representations
        out_teacher, teacher_visual_features, _ = self.teacher.forward_output_logits(x, y)

        # All to do feature distillation
        stacked_activations = []
        # Iterate over batches of activations
        for batch in self.student_decoder_activations.values():
            # Each `batch` is a list of tensors; directly stack them
            stacked_batch = torch.stack(batch)  # This stacks along a new dimension, resulting in [Batch, layers, x, y] for each batch
            stacked_activations.append(stacked_batch)
        # Optionally, if you have multiple batches and want to concatenate them along the batch dimension
        student_final_activations = torch.cat(stacked_activations, dim=0)

        # LOSS 1: Get feature maps and match and compute loss
        # Student activations
        student_encoder_fmaps = [torch.mean(fmap, dim=[2, 3]) for fmap in student_image_enc_fmaps]
        # Teacher activations (get cls token)
        teacher_encoder_fmaps = [torch.stack(fmap)[:, 0, ...].squeeze() for fmap in self.teacher_encoder_activations.values()]
        teacher_encoder_fmaps = [fmap.reshape(-1, 1024) for fmap in teacher_encoder_fmaps]
        # Project student fmaps to teacher dimensionality
        student_encoder_fmaps = [self.student.projectors[i](fmap) for i, fmap in enumerate(student_encoder_fmaps)]
        # Compute feature map distillation loss
        fmap_loss = self.fmap_distill_loss(torch.stack(teacher_encoder_fmaps).to(device), torch.stack(student_encoder_fmaps).to(device))

        # LOSS 2: Compute a loss between output logits of teacher and student
        # Student logits shape: [B, GT_length, vocab]
        # Teacher logits shape: [B, GT_length, vocab]
        temperature = 1
        teacher_logits = torch.cat(out_teacher, dim=0).to(device)
        student_logits = student_decoder_output
        teacher_logits_kl = teacher_logits/temperature
        student_logits_kl = student_logits/temperature
        kl_loss = self.kl_div_loss(student_logits_kl.log_softmax(dim=-1), teacher_logits_kl.softmax(dim=-1))
        kl_loss = kl_loss*(temperature ** 2)

        # LOSS 3: Compute loss between output of student and GT
        y_target = y[:, 1:].reshape(-1)
        y_target = y_target.view(-1)
        y_pred = student_decoder_output[:, :-1].view(-1, student_decoder_output.size(-1))
        ce_loss = self.ce_loss(y_pred, y_target)

        # LOSS 4: Compute loss between final encoding of Student and Teacher
        # final_enc_loss=self.final_encoding_loss(student_visual_features,teacher_visual_features)
        # teacher_visual_features= torch.stack(teacher_visual_features,dim=0)
        # print(memory.shape)
        # print(teacher_visual_features.shape)

        # LOSS 5: Tried but didn't work:
        # Cross entropy between teacher targets and Student predictions
        # Get the output of teacher inference
        # out_teacher_inference=self.teacher(x)
        # n_student_tokens=student_logits.shape[1]
        # out_teacher_inference_targets=[]
        # for i in out_teacher_inference:
            # teacher_tokens = i['predictions'][0].tolist()

            # Ensure the teacher's sequence is the same length as the student's
            # adjusted_teacher_tokens = teacher_tokens[:n_student_tokens]  # Truncate if necessary
            # adjusted_teacher_tokens += [102] * (n_student_tokens - len(adjusted_teacher_tokens))  # Pad if necessary

            # out_teacher_inference_targets.append(torch.tensor(adjusted_teacher_tokens))

        # Stack adjusted target sequences and ensure the tensor is on the same device as the logits
        # out_teacher_inference_targets = torch.stack(out_teacher_inference_targets).to(student_logits.device)
        # Compute the loss
        # ce_loss_2 = self.ce_loss2(student_logits.view(-1, student_logits.size(-1)),out_teacher_inference_targets.view(-1))


        # Loss 6: Decoder Feature Distillation
        # Using the teacher decoder activations instead
        teacher_distill = []
        for activations in self.teacher_decoder_activations.values():
            stacked_activations = torch.stack(activations, dim=0)
            teacher_distill.append(stacked_activations.squeeze(1))
        teacher_distill = torch.stack(teacher_distill,dim=0)
        swapped_tensor = teacher_distill.permute(1, 0, 2, 3)
        teacher_decoder_distill = swapped_tensor[:, [0, 2, 3, 5], 1542:, :]

        B,L,S,E=student_final_activations.shape
        student_decoder_distill_proj = student_final_activations.view(-1, E)
        student_decoder_distill=self.student.project_decoder(student_decoder_distill_proj)
        student_decoder_distill = student_decoder_distill.view(B, L, S, -1)
        student_decoder_distill=student_decoder_distill.permute(1, 0, 2, 3)

        # decoder_loss=self.decoder_distill_loss(teacher_decoder_distill,student_decoder_distill)
        
        # Our final Loss function currently looks at Loss 3
        loss = ce_loss + kl_loss + fmap_loss

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_fmap_loss", fmap_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Inactive Loss Functions
        # self.log("train_ce_loss_2", ce_loss_2, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("train_enc_loss", final_enc_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("train_decoder_loss", decoder_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("train_fmap_loss", fmap_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Clear the teacher activations for this batch
        del self.teacher_encoder_activations
        del self.student_decoder_activations
        del self.teacher_decoder_activations
        self.teacher_encoder_activations = {}
        self.student_decoder_activations = {}
        self.teacher_decoder_activations = {}

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, caption_id,vid_id = batch['frames'], batch['caption'], batch['caption-id'],batch['vid-id']

        # We are performing both decoders to see which has better performance
        student_decoder_output = self.student.greedy_decode(x, max_len=y.shape[-1]+5)

        # Decoding the greedy prediction from student using teacher tokenizer
        preds = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in student_decoder_output]

        # Decoding the captions in Ground Truth with tokenizer
        caps = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in y]

        # Add BLEU for student
        caps = [[c] for c in caps]

        loss = metrics.calculate_bleu_score_corpus(caps, preds)

        print(f'Ground-Truth Captions: {caps}')
        print(f'Student Predictions: {preds}')
        print(f'BLEU@4: {loss}')

        with open(self.logpath + '/' + self.filename, 'a') as f:
            f.write("\n" * 2)
            f.write("Validation Results\n")
            f.write(f'Epoch: {self.current_epoch}\n')
            f.write(f'Ground-Truth Captions: {caps}\n')
            f.write(f'Student Predictions: {preds}\n')
            f.write(f'BLEU@4: {loss}\n')
        
        self.log("val_loss", loss, prog_bar=True)

        for i in range(0,len(preds)):
            self.validation_step_outputs.append({
            "image_id": str(vid_id[i]),  # Make sure this is just an integer, not a tensor
            "caption": preds[i]
        })
            
        del self.teacher_encoder_activations
        del self.student_decoder_activations
        del self.teacher_decoder_activations
        self.teacher_encoder_activations = {}
        self.student_decoder_activations = {}
        self.teacher_decoder_activations = {}

        # Optionally, you can save the results to a file here,
        # or you can return them and collect them in `validation_epoch_end`
        return loss
    
    def on_validation_epoch_end(self):
        # We call metrics which has a function to calculate BLEU-4, Rouge, Cider, and Meteor
        metrics.calculate_score(self.validation_step_outputs, self.logpath + '/' + 'validation_' + self.filename, self.logpath)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, caption_id,vid_id = batch['frames'], batch['caption'], batch['caption-id'],batch['vid-id']

        # We are performing both decoders to see which has better performance
        student_decoder_output = self.student.greedy_decode(x, max_len=y.shape[-1]+5)

        # Decoding the greedy prediction from student using teacher tokenizer
        preds = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in student_decoder_output]

        # Decoding the captions in Ground Truth with tokenizer
        caps = [self.teacher.tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in y]

        # Add BLEU for student
        caps = [[c] for c in caps]
        
        loss = metrics.calculate_bleu_score_corpus(caps, preds)

        with open(self.logpath + '/' + self.filename, 'a') as f:
            f.write("\n" * 2)
            f.write("Test Results\n")
            f.write(f'Epoch: {self.current_epoch}\n')
            f.write(f'Ground-Truth Captions: {caps}\n')
            f.write(f'Student Predictions: {preds}\n')

        self.log("test_loss", loss, prog_bar=True)

        for i in range(0, len(preds)):
            self.test_step_outputs.append({
            "image_id": str(vid_id[i]),  # Make sure this is just an integer, not a tensor
            "caption": preds[i]
        })

        del self.teacher_encoder_activations
        del self.student_decoder_activations
        del self.teacher_decoder_activations
        self.teacher_encoder_activations = {}
        self.student_decoder_activations = {}
        self.teacher_decoder_activations = {}
        return loss
    
    def on_test_epoch_end(self):
        # We call metrics which has a function to calculate BLEU-4, Rouge, Cider, and Meteor
        metrics.calculate_score(self.test_step_outputs, self.logpath + '/' + 'test_' + self.filename, self.logpath)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=4,
                                                                  min_lr=1e-8,
                                                                  factor=0.5)
        total_steps = self.epochs * self.steps
        scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor": "val_loss"}]

    def get_teacher_encoder_activation(self, name):
        def hook(model, input, output):
            if name in self.teacher_encoder_activations:
                self.teacher_encoder_activations[name].append(output.detach())
            else:
                self.teacher_encoder_activations[name] = [output.detach()]

        return hook
    
    def get_student_decoder_activation(self, name):
        def hook(model, input, output):
            if name in self.student_decoder_activations:
                self.student_decoder_activations[name].append(output.detach())
            else:
                self.student_decoder_activations[name] = [output.detach()]

        return hook
    
    def get_teacher_decoder_activation(self, name):
        def hook(model, input, output):
            if name in self.teacher_decoder_activations:
                self.teacher_decoder_activations[name].append(output.detach())
            else:
                self.teacher_decoder_activations[name] = [output.detach()]

        return hook

    def teardown(self, stage: str) -> None:
        for hook in self.teacher_encoder_hooks:
            hook.remove()

        for hook in self.teacher_decoder_hooks:
            hook.remove()

        for hook in self.student_decoder_hooks:
            hook.remove()
