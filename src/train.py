'''
TRAINING SCRIPT
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple,List
import pandas as pd

import lightning as L
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS
from utils.dataloader import CaptionDataset
from transformers import BertTokenizer

from config import cfg
from src.models.model import StudentCandidate
from lightning.pytorch.loggers import WandbLogger

os.environ['WANDB_MODE'] = cfg['WANDB']['MODE']


def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(train_paths: Tuple[str, str],train_ids:List[str], train_data: pd.DataFrame, val_paths: Tuple[str, str],val_ids:List[str], test_data: pd.DataFrame, model_args: dict,
          log_args:dict, chkpt_args:dict, trainer_args: dict, batch_size:int):
    """Train a model.

    Args:
        data_path: Path to data files
        model_args: Dictionary of kwargs for model
        trainer_args: Dictionary of kwargs for Trainer

    Returns:
        Trained model instance.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # ---------------

    # WANDB Logger
    wandb_logger = WandbLogger(project="real-time-video-captioning")

    train_dataset = CaptionDataset(*train_paths, vid_ids=train_ids,data=train_data, tokenizer=tokenizer)
    val_dataset = CaptionDataset(*val_paths, vid_ids=val_ids,data=test_data,tokenizer=tokenizer)

    # data loader  
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Instantiate the model
    model = StudentCandidate(**model_args)
    checkpoint = ModelCheckpoint(**chkpt_args)

    # log gradients and model topology
    # wandb_logger.watch(model)
    logger = wandb_logger
    # logger = TensorBoardLogger(**log_args)

    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args, callbacks=checkpoint, logger=logger)
    
    # Fit the model
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    wandb_logger.experiment.unwatch(model)


if __name__ == "__main__":
    # Organize arguments here
    train_args = cfg['TRAIN']
    model_def = train_args['model_def']

    # with open(cfg['DATA']['CAPTIONS_PATH'], 'r') as file:
    #     data = json.load(file)
    # annotations_list = data['annotations']
    # annotations_df = pd.DataFrame(annotations_list)

    df = pd.read_csv(cfg['DATA']['CAPTIONS_PATH'])
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    with open(cfg['DATA']['TRAIN_IDS'], 'r') as file:
        train_ids = [line.strip() for line in file]

    with open(cfg['DATA']['VAL_IDS'], 'r') as file:
        val_ids = [line.strip() for line in file]

    model_instance_args = cfg['MODEL'][model_def]
    chkpt_args = cfg['CALLBACK']
    trainer_args = train_args['TRAINER']
    LR = train_args['LR']
    train_data_path = cfg['DATA']['TRAIN_PATH']
    train_ids=cfg['DATA']['TRAIN_IDS']
    val_data_path = cfg['DATA']['VAL_PATH']
    model_args = {'model_args': model_instance_args,
                  'lr': LR}
    log_args = cfg['LOGGER']
    batch_size = train_args['BATCH_SIZE']

    # Train the model
    train(train_data_path,train_ids,train_df,val_data_path,val_ids,test_df, model_args, log_args, chkpt_args, trainer_args, batch_size=batch_size)
    