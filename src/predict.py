import torch
from GenerativeImage2Text.generativeimage2text.model import get_git_model
from GenerativeImage2Text.generativeimage2text.torch_common import load_state_dict
from GenerativeImage2Text.generativeimage2text.tsv_io import load_from_yaml_file
from GenerativeImage2Text.generativeimage2text.process_image import load_image_by_pil
from GenerativeImage2Text.generativeimage2text.inference import get_image_transform
from transformers import BertTokenizer
import logging

def perform_inference(model, prefix, param, img_path):
    if isinstance(img_path, str):
        img_path = [img_path]
    img = [load_image_by_pil(i) for i in img_path]
    transforms = get_image_transform(param)
    img = [transforms(i) for i in img]
    img = [i.unsqueeze(0) for i in img]
    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    with torch.no_grad():
        result = model({
            'image': img,
            'prefix': torch.tensor(input_ids).unsqueeze(0),
        })
    
    cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
    print(cap)
    #These are the visual features, it is either for one image, or up to a max of 6 concatenated features for a series of images. Final output of encoder.
    print(result['visual_features'].shape)
    #This is a dictionary of dictionary of tensors. First index of dictionary is index of token in the caption. It starts from 1. Second index is word token indices. There are many
    #Candidates for each word token. You can find the logits related to the actual word predicted, but taking the word index from the caption.
    print(result['logits_dict'][1][1037].shape)
    logging.info('output: {}'.format(cap))


if __name__ == "__main__":
    model_path = 'results/model.pt'
    model_name = 'GIT_LARGE_MSRVTT'
    ckpt = torch.load(model_path)
    ckpt = ckpt['model']

    param = load_from_yaml_file(f'GenerativeImage2Text/aux_data/models/{model_name}/parameter.yaml')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = get_git_model(tokenizer, param)
    #print(model)
    model.eval()
    load_state_dict(model, ckpt)
    #img_path = 'GenerativeImage2Text/aux_data/images/1.jpg'
    img_path = ['GenerativeImage2Text/aux_data/images/golf-swing/0.png','GenerativeImage2Text/aux_data/images/golf-swing/1.png','GenerativeImage2Text/aux_data/images/golf-swing/2.png']
    #img_path = ['GenerativeImage2Text/aux_data/images/golf-swing/0.png','GenerativeImage2Text/aux_data/images/golf-swing/1.png','GenerativeImage2Text/aux_data/images/golf-swing/2.png','GenerativeImage2Text/aux_data/images/golf-swing/3.png','GenerativeImage2Text/aux_data/images/golf-swing/4.png',
    #            'GenerativeImage2Text/aux_data/images/golf-swing/5.png','GenerativeImage2Text/aux_data/images/golf-swing/6.png','GenerativeImage2Text/aux_data/images/golf-swing/7.png','GenerativeImage2Text/aux_data/images/golf-swing/8.png','GenerativeImage2Text/aux_data/images/golf-swing/9.png']
    perform_inference(model, '', param, img_path)
    print('Done')
