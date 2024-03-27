import pandas as pd
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose, ToPILImage
from PIL import Image
from typing import Dict, Optional, List, Any
from src.utils.video_handlers import get_video_frames, get_evenly_sampled_frames
from src.utils.tokenizer import encode_caption
from transformers import PreTrainedTokenizer
import time

def image_transform():
    crop_size = 224
    trans = [
        ToTensor(),
        Resize(crop_size, interpolation=Image.BICUBIC),
        CenterCrop(crop_size),
        lambda image: image[[2, 1, 0], ...],  # BGR -> RGB
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
    ]
    transforms = Compose(trans)

    return transforms


class CaptionDataset(Dataset):
    def __init__(self, data_path: str, vid_ids: List[str], data: pd.DataFrame, encoded_caption_data: Dict,
                 transform: Optional = None, num_frames: int = 10, random_state: Optional = None):
        """Dataset for loading videos with respective captions as ground-truth labels.

        Args:
            data_path: String, path to videos
            vid_ids: List of video ids (each are a data point)
            data: DataFrame matching captions to video ids
            encoded_caption_data: Dictionary containing tokenizer/encoded captions
            transform: Transformations to apply to videos
            num_frames: Number of frames to keep
            random_state: Integer for seeding caption selection
        """
        self.data_path = data_path
        self.vid_ids = vid_ids
        self.data = data
        self.transform = transform if transform else image_transform()
        self.num_frames = num_frames
        self.random_state = random_state
        self.encoded_caption_data = encoded_caption_data

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        # Get video id and its captions
        s1 = time.time()
        vid_id = self.vid_ids[idx]
        caption_ids = self.data.loc[self.data['image_id'] == vid_id, 'id']

        # Select a random caption to use as label
        caption_id = caption_ids.sample(n=1, random_state=self.random_state).iloc[0]

        # Encode the caption
        # encoded_caption = encode_caption(caption, self.tokenizer)
        encoded_caption = self.encoded_caption_data[caption_id]
        s2 = time.time()

        s3 = time.time()
        # Load video frames into list
        vid_path = os.path.join(self.data_path, vid_id + '.mp4')
        # raw_frames = get_video_frames(vid_path)
        raw_frames = get_evenly_sampled_frames(vid_path, self.num_frames)
        s4 = time.time()

        s5 = time.time()
        # Sample frames from all frames and transform only those
        raw_frames = raw_frames[torch.arange(0, raw_frames.shape[0], raw_frames.shape[0] // self.num_frames)[:self.num_frames]]
        frames = torch.stack([self.transform(frame) for frame in raw_frames])
        s6 = time.time()

        first = s2 - s1
        second = s4-s3
        third = s6-s5

        # frames shape: [N, C, 224, 224], caption shape: [?]
        return {'frames': frames, 'caption': encoded_caption}


def collate_fn(batch: Any):
    """Collate function to handle variable sized captions tensors during data loading.
    Inspired from:
    https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/data_layer/builder.py

    Args:
        batch: Output batch of elements from the Dataset class

    Returns:
        Collated batch for use in model ingestion
    """
    element = batch[0]
    if isinstance(element, dict):  # Maintain dictionary structure
        return {key: collate_fn([d[key] for d in batch]) for key in element}
    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            # Handle mismatched shapes (ex. one tensor shorter/longer than another in minibatch)
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                # Find longest caption tensor
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if b.shape < max_shape:
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        b2[:b.shape[0]] = b
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)
