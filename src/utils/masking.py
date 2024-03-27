import torch


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

