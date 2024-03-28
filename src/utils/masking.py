import torch


def create_cls_memory_mask(memory_length: int, cls_token_index: int = 0):
    """
    Create a memory mask that allows attending only to the CLS token in the memory.
    
    Args:
        memory_length: The length of the memory sequence.
        cls_token_index: The index of the CLS token in the memory sequence.
        
    Returns:
        A memory mask tensor of shape (1, 1, memory_length) where all positions
        are masked except for the CLS token position.
    """
    mask = torch.ones((1, 1, memory_length), dtype=torch.bool)
    mask[:, :, cls_token_index] = False 
    return mask

def create_padding_mask(seq: torch.Tensor, padding_token: int = 0):
    """ Generate a mask for padding indices.

    Args:
        seq: Padded sequence
        padding_token: Token for padded values

    Returns:
        Masked tensor indicated indices of padded tokens
    """
    return seq == padding_token


def create_casual_mask(size: int):
    """ Generate a causal mask of specified size.

    Args:
        size: Number of rows and columns

    Returns:
        A size x size causal mask (upper triangular)
    """
    return torch.triu(torch.ones(size, size), diagonal=1).bool()

