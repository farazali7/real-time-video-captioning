"""
Inference Script:
- Ensure that the student and/or teacher architecture is defined in the config file correctly to match the loaded weights.
- Load the student and teacher models from the checkpoint. Weights are loeaded based on the run_name, so ensure that the run_name is correct.
- Load the test data and perform inference on the test data.
- Print the ground truth captions and the predicted captions.
"""

import os, glob
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import pickle

from config import cfg
from src.models.model import StudentCandidateV1, GenerativeImageTextTeacher
from .utils.dataloader import CaptionDataset, collate_fn

def inference(data_args, model_args, teacher_model_args, run_name, device):
    dataset = CaptionDataset(**data_args)
    dataloader = DataLoader(dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False,
                            pin_memory=True, num_workers=4, collate_fn=collate_fn, persistent_workers=True)

    student_model = StudentCandidateV1(**model_args).to(device)
    teacher_model = GenerativeImageTextTeacher(**teacher_model_args).to(device)
    
    model_path = glob.glob(os.path.join("results", "run", run_name, "*.ckpt"))
    if not model_path:
        raise FileNotFoundError(f"No checkpoint found in 'results/run/{run_name}'")
    model_path = model_path[0]
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    student_state_dict = {k.replace("student.", ""): v for k, v in state_dict.items() if k.startswith("student.")}
    student_model.load_state_dict(student_state_dict)

    student_model.eval()
    teacher_model.eval()

    tokenizer = teacher_model.tokenizer

    for batch in dataloader:
        x, y, _, _ = batch['frames'], batch['caption'], batch['caption-id'], batch['vid-id']
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            student_decoder_output = student_model.greedy_decode(x, max_len=y.shape[-1] + 5)
            preds = [tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in student_decoder_output]

        caps = [[tokenizer.decode(c.tolist(), skip_special_tokens=True)] for c in y]

        print(f"Ground Truth: {caps}")
        print(f"Predictions: {preds}")
        break

if __name__ == "__main__":
    """
    Inference:
    - This will only work if there is a model checkpoint stored in the 'results/run/run_name' directory.
    - The run_name should be passed as an argument.
    - The student and teacher model architecture should be defined in the config file.
    - We only load student weights.
    - The test data is loaded from the data_path and captions_path.
    - The encoded caption data is loaded from the encoded_caption_ids path.
    - The test data is loaded and inference is performed.
    - The ground truth and predicted captions are printed.

    Usage:
    >>> pwd
    real-time-video-captioning/
    >>> python3 -m src.inference run_name
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.inference run_name")
        sys.exit(1)
    run_dir = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference will be performed on {device}.")

    data_path, df_path, encoded_captions_path = cfg['DATA']['VIDEOS_PATH'], cfg['DATA']['CAPTIONS_PATH'], cfg['DATA']['ENCODED_CAPTION_IDS']
    df = pd.read_csv(df_path)
    with open(encoded_captions_path, 'rb') as f:
        encoded_caption_data = pickle.load(f)

    test_data = df[df['split'] == 'test']
    test_ids = test_data['image_id'].unique().tolist()

    test_data_args = {'data_path': data_path, 'vid_ids': test_ids, 'data': test_data, 'encoded_caption_data': encoded_caption_data}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    vocab_length = len(tokenizer.vocab)

    student_model_def = cfg['TRAIN']['STUDENT_MODEL_DEF']
    model_args = {**cfg['MODEL'][student_model_def], 'vocab_length': vocab_length, 'cls_token_id': tokenizer.cls_token_id, 'sep_token_id': tokenizer.sep_token_id}

    teacher_model_def = cfg['TRAIN']['TEACHER_MODEL_DEF']
    teacher_model_args = cfg['MODEL'][teacher_model_def]

    inference(test_data_args, model_args, teacher_model_args, run_dir, device)