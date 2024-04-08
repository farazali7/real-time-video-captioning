"""
Testing Script

This script is used for testing models, inference and saving student, in case not done in training.
"""

import os
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import lightning as L
import torch.cuda
import torch.utils.data
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import BertTokenizer
import pickle

from config import cfg
from src.models.model import StudentCandidateV1, GenerativeImageTextTeacher, DistillationTrainer
from .utils.dataloader import CaptionDataset, collate_fn


def test(  
        test_data_args: Dict,
        student_model_args: Dict, 
        teacher_model_args: Dict, 
        trainer_args: Dict, 
        batch_size: int, 
        lr: float,
        path:str) -> DistillationTrainer:
    """
    Test function for knowledge distillation experiments.

    Args:
        test_data_args (Dict): Dictionary of test dataset arguments.
        student_model_args (Dict): Dictionary of student model instance arguments.
        teacher_model_args (Dict): Dictionary of teacher model instance arguments.
        trainer_args (Dict): Dictionary of PyTorch Lightning Trainer arguments.
        batch_size (int): Batch size for the training.
        lr (float): Learning rate for the optimizer.

    Returns:
        DistillationTrainer: The trained distillation model instance.
    """
    
    test_dataset = CaptionDataset(
        **test_data_args
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    student_model = StudentCandidateV1(**student_model_args)
    teacher_model = GenerativeImageTextTeacher(**teacher_model_args)
    model=DistillationTrainer.load_from_checkpoint(checkpoint_path=path,teacher=teacher_model,
        student=student_model,
        lr=lr,
        steps=len(test_dl),
        epochs=trainer_args['max_epochs'])
    
    #student_model=model.student
    #torch.save(student_model,"results/student_model.pt")
   
    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer()
    
    # Optionally, perform testing
    trainer.test(model=model, dataloaders=test_dl)

    return model


if __name__ == "__main__":
    # Load configuration
    train_args = cfg['TRAIN']

    # Dataset preparation
    data_path = cfg['DATA']['VIDEOS_PATH']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    random_state = cfg['SEED']

    # Load and prepare caption data
    df = pd.read_csv(cfg['DATA']['CAPTIONS_PATH'])
    with open(cfg['DATA']['ENCODED_CAPTION_IDS'], 'rb') as f:
        encoded_caption_data = pickle.load(f)

    # Data splits
    train_data, val_data, test_data = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_ids, val_ids, test_ids = train_data['image_id'].unique().tolist(), val_data['image_id'].unique().tolist(), test_data['image_id'].unique().tolist()

    train_data_args = {
        'data_path': data_path,
        'vid_ids': train_ids,
        'data': train_data,
        'encoded_caption_data': encoded_caption_data,
        'random_state': random_state
    }

    val_data_args = {
        'data_path': data_path,
        'vid_ids': val_ids,
        'data': val_data,
        'encoded_caption_data': encoded_caption_data,
        'random_state': random_state
    }

    test_data_args = {
        'data_path': data_path,
        'vid_ids': test_ids,
        'data': test_data,
        'encoded_caption_data': encoded_caption_data,
        'random_state': random_state
    }

    vocab_length = len(tokenizer.vocab)
    
    student_model_def, teacher_model_def = train_args['STUDENT_MODEL_DEF'], train_args['TEACHER_MODEL_DEF']
    
    student_model_args = {
        **cfg['MODEL'][student_model_def],
        'vocab_length': vocab_length,
        'cls_token_id': tokenizer.cls_token_id,
        'sep_token_id': tokenizer.sep_token_id
    }

    teacher_model_args = cfg['MODEL'][teacher_model_def]

    callback_args = cfg['CALLBACK']

    trainer_args = {
        **train_args['TRAINER'],
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else 1
    }

    if not torch.cuda.is_available():
        trainer_args['precision'] = '32-true'  # Avoid bfloat16 precision if training on CPU

    batch_size, lr = train_args['BATCH_SIZE'], train_args['LR']

    test(test_data_args,student_model_args,teacher_model_args,trainer_args,batch_size,lr,"results/all_4_losses.ckpt")