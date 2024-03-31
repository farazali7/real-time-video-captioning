'''
TRAINING SCRIPT
'''

import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd

import lightning as L
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
from .utils.dataloader import CaptionDataset, collate_fn
from transformers import BertTokenizer
import pickle

from config import cfg
from src.models.model import StudentCandidateV1, GenerativeImageTextTeacher, DistillationTrainer
from lightning.pytorch.loggers import WandbLogger

os.environ['WANDB_MODE'] = cfg['WANDB']['MODE']


def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(train_data_args: Dict, val_data_args: Dict,
          student_model_args: Dict, teacher_model_args: Dict,
          callback_args: Dict, trainer_args: Dict, batch_size: int, lr: float):
    """ Training function for knowledge distillation experiments

    Args:
        train_data_args: Dictionary of train dataset arguments
        val_data_args: Dictionary of val dataset arguments
        student_model_args: Dictionary of student model instance arguments
        teacher_model_args: Dictionary of teacher model instance arguments
        callback_args: Dictionary of training callback arguments
        trainer_args: Dictionary of PyTorch Lightning Trainer arguments
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Trained model instance.
    """
    # WANDB Logger
    wandb_logger = WandbLogger(project="real-time-video-captioning")

    # Create datasets and dataloaders
    train_dataset = CaptionDataset(**train_data_args)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                          collate_fn=collate_fn)

    val_dataset = CaptionDataset(**val_data_args)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                        collate_fn=collate_fn)

    # Instantiate the student and teacher models and pass to Lightning module
    student_model = StudentCandidateV1(**student_model_args)
    teacher_model = GenerativeImageTextTeacher(**teacher_model_args)
    distillation_model = DistillationTrainer(teacher=teacher_model, student=student_model, lr=lr,steps=len(train_dl),epochs=trainer_args['max_epochs'])

    callback = ModelCheckpoint(**callback_args)

    # log gradients and model topology
    logger = wandb_logger
    # logger = TensorBoardLogger(**log_args)

    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args, callbacks=callback, logger=logger, num_sanity_val_steps=0)
    # Fit the model
    trainer.fit(model=distillation_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    wandb_logger.experiment.unwatch(distillation_model)

    return distillation_model


if __name__ == "__main__":
    # Organize arguments here
    train_args = cfg['TRAIN']

    # Dataset arguments
    data_path = cfg['DATA']['VIDEOS_PATH']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    random_state = cfg['SEED']

    df = pd.read_csv(cfg['DATA']['CAPTIONS_PATH'])
    with open(cfg['DATA']['ENCODED_CAPTION_IDS'], 'rb') as f:
        encoded_caption_data = pickle.load(f)

    train_data = df[df['split'] == 'train']
    train_ids = train_data['image_id'].unique().tolist()
    train_data_args = {'data_path': data_path,
                       'vid_ids': train_ids,
                       'data': train_data,
                       'encoded_caption_data': encoded_caption_data,
                       'random_state': random_state}

    val_data = df[df['split'] == 'val']
    val_ids = val_data['image_id'].unique().tolist()
    val_data_args = {'data_path': data_path,
                     'vid_ids': val_ids,
                     'data': val_data,
                     'encoded_caption_data': encoded_caption_data,
                     'random_state': random_state}

    test_data = df[df['split'] == 'test']
    test_ids = test_data['image_id'].unique().tolist()
    test_data_args = {'data_path': data_path,
                      'vid_ids': test_ids,
                      'data': test_data,
                      'encoded_caption_data': encoded_caption_data,
                      'random_state': random_state}

    # Model arguments
    vocab_length = len(tokenizer.vocab)
    student_model_def = train_args['STUDENT_MODEL_DEF']
    student_model_args = cfg['MODEL'][student_model_def]
    student_model_args['vocab_length'] = vocab_length
    student_model_args['cls_token_id'] = tokenizer.cls_token_id
    student_model_args['sep_token_id'] = tokenizer.sep_token_id

    teacher_model_def = train_args['TEACHER_MODEL_DEF']
    teacher_model_args = cfg['MODEL'][teacher_model_def]

    # Checkpoint arguments
    callback_args = cfg['CALLBACK']

    # Trainer arguments
    trainer_args = train_args['TRAINER']
    is_accelerator_available = torch.cuda.is_available()
    trainer_args['accelerator'] = 'gpu' if is_accelerator_available else 'cpu'
    trainer_args['devices'] = torch.cuda.device_count() if is_accelerator_available else 1
    if not is_accelerator_available:  # If on CPU then avoid bfloat16 (used for GPUs) issue
        trainer_args['precision'] = '32-true'

    batch_size = train_args['BATCH_SIZE']
    lr = train_args['LR']

    # Train the model
    train(train_data_args, val_data_args, student_model_args, teacher_model_args,
          callback_args, trainer_args, batch_size, lr)
