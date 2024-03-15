'''
CONFIGURATION SETTINGS FOR PROJECT
'''
import torch

cfg = {
    'DATA': {
        'TRAIN_PATH': 'data/train_videos',
        'VAL_PATH': 'data/test_videos',
        'TRAIN_IDS': 'data/train_ids.txt',
        'VAL_IDS': 'data/test_ids.txt',
        'CAPTIONS_PATH': 'data/labels.csv',
    },
    'CALLBACK': {
        'dirpath': 'results/',
        'save_top_k': 2,
        'monitor': 'val_loss'
    },
    'LOGGER': {
        'save_dir': 'results/',
        'name': 'captions'
    },
    'TRAIN': {
        'model_def': 'StudentCandidate',
        'TRAINER': {
            'max_epochs': 100,
            'precision': 16,
            'enable_checkpointing': True,
            'accelerator': 'gpu',
            'devices': int(torch.cuda.device_count()),
            'strategy': 'ddp_find_unused_parameters_true'
        },
        'LR': 1e-5,
        'BATCH_SIZE': 16,
    },
    'MODEL': {
        'StudentCandidate': {},
    },
    'WANDB': {
        "MODE": 'online'  # One of {'online', 'offline', 'disabled'}
    }
}