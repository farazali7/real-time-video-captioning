import torch
from generativeimage2text.model import get_git_model
from generativeimage2text.torch_common import load_state_dict
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.process_image import load_image_by_pil
from generativeimage2text.inference import get_image_transform
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
    print("Caption:", cap)
    logging.info('output: {}'.format(cap))


if __name__ == "__main__":
    model_path = 'results/model.pt'
    model_name = 'GIT_LARGE_MSRVTT'
    ckpt = torch.load(model_path)
    ckpt = ckpt['model']

    param = load_from_yaml_file(f'GenerativeImage2Text/aux_data/models/{model_name}/parameter.yaml')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = get_git_model(tokenizer, param)
    model.eval()
    load_state_dict(model, ckpt)

    img_path = 'GenerativeImage2Text/aux_data/images/2.jpg'
    perform_inference(model, '', param, img_path)

    print('Done')
