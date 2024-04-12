'''
Configuration File:
    - SEED: Random seed for reproducibility
    - DATA: Data paths
        - VIDEOS_PATH: Path to the videos
        - CAPTIONS_PATH: Path to the captions
        - ENCODED_CAPTION_IDS: Path to the encoded captions
    - CALLBACK: Model checkpointing
        - dirpath: Directory to save the model checkpoints
        - filename: Model checkpoint filename
        - save_top_k: Number of checkpoints to save
        - monitor: Metric to monitor
        - mode: Mode to monitor
    - LOGGER: Logging
        - save_dir: Directory to save the logs
        - name: Name of the log file
    - TRAIN: Training configuration
        - STUDENT_MODEL_DEF: Student model definition
        - TEACHER_MODEL_DEF: Teacher model definition
        - TRAINER: PyTorch Lightning Trainer configuration
        - LR: Learning rate
        - BATCH_SIZE: Batch size
    - MODEL: Model configuration
        - StudentCandidateV1: Student model configuration
            - image_enc_name: Image encoder name
            - d_model: Model dimension
            - n_head: Number of heads in the multi-head attention
            - d_ffn: Feed forward dimension
            - dropout: Dropout rate
            - num_decoder_layers: Number of decoder layers
        - GenerativeImageTextTeacher: Teacher model configuration
            - param_path: Path to the teacher model configuration
            - pretrained_weights: Path to the teacher model weights
    - WANDB: Weights and Biases configuration
        - MODE: Online or offline mode
'''


cfg = {
    'SEED': 5,
    'DATA': 
    {
        'VIDEOS_PATH': 'data/MSRVTT/videos/all',
        'CAPTIONS_PATH': 'data/labels/labels.csv',
        'ENCODED_CAPTION_IDS': 'data/labels/encoded_captions.pkl'
    },
    'CALLBACK': 
    {
        'dirpath': 'results/',
        'filename': 'model-{epoch:02d}',
        'save_top_k': 1,
        'monitor': 'epoch',
        'mode': 'max',
    },
    'LOGGER': 
    {
        'save_dir': 'results/',
        'name': 'captions'
    },
    'TRAIN': 
    {
        'STUDENT_MODEL_DEF': 'StudentCandidateV1',
        'TEACHER_MODEL_DEF': 'GenerativeImageTextTeacher',
        'TRAINER': 
        {
            'max_epochs': 20,
            'precision': 16,
            'enable_checkpointing': True,
            'strategy': 'auto'
        },
        'LR': 1e-4,
        'BATCH_SIZE': 8,
    },
    'MODEL': 
    {
        'StudentCandidateV1': 
        {
            'image_enc_name': 'tiny_vit_21m_224.dist_in22k_ft_in1k',
            'd_model': 576,
            'n_head': 8,
            'd_ffn': 1024,
            'dropout': 0.3,
            'num_decoder_layers': 2
        },
        'GenerativeImageTextTeacher': 
        {
            'param_path': 'data/teacher_configs/GIT_LARGE_MSRVTT/parameter.yaml',
            'pretrained_weights': 'results/model.pt'
        }
    },
    'WANDB': 
    {
        "MODE": 'online'
    }
}