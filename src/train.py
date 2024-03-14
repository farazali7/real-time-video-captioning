# This is for training the model
from typing import List, Optional, Union, Tuple
import time
from tqdm import tqdm
import torch
from config import cfg
from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

#import dataloader

def train(model_args: dict,log_args:dict, chkpt_args:dict, trainer_args: dict, batch_size:int, data_perc: float,optimizer:torch.optim.Optimizer):
    """Train a model.

    Args:
        data_path: Path to data files
        model_args: Dictionary of kwargs for model
        trainer_args: Dictionary of kwargs for Trainer

    Returns:
        Trained model instance.
    """
    #Create Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])

    # WANDB Logger
    wandb_logger = WandbLogger(project="multimodal_xray")

    train_dataset = Dataset(transform=transform, perc=data_perc)
    val_dataset = Dataset(transform=transform, perc=data_perc)

    # data loader  
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Instantiate the model
    model = Model(**model_args)
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
    model_instance_args = cfg['MODEL'][model_def]
    chkpt_args = cfg['CALLBACK']
    trainer_args = train_args['TRAINER']
    LR = train_args['LR']
    data_perc = train_args['DATA_PERC']
    model_args = {'model_args': model_instance_args,
                  'lr': LR}
    log_args = cfg['LOGGER']
    batch_size = train_args['BATCH_SIZE']

    # Train the model
    train( model_args, log_args, chkpt_args, trainer_args, batch_size=batch_size, data_perc=data_perc)