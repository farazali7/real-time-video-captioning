import torch
from torch.nn import Module
from typing import Dict

from src.models.model import StudentCandidateV1


def load_kd_student_model(ckpt_path: str, model_args: Dict) -> Module:
    """
    Load a trained, knowledge distilled student model and remove all projector parameters present.
    Args:
        ckpt_path: Path to model checkpoint
        model_args: Dictionary of keyword arguments for constructor

    Returns:
        Student model with no auxiliary projector layers
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    student_state_dict = {k.replace("student.", ""): v for k, v in checkpoint['state_dict'].items() if
                          k.startswith("student.")}
    student = StudentCandidateV1(**model_args)
    student.load_state_dict(student_state_dict)

    # Delete projectors (these are irrelevant for inference and usually have
    # a LazyMixin layer which causes issues with model metadata inquiries
    del student.project_decoder
    del student.project
    del student.upsample
    del student.projectors

    return student
