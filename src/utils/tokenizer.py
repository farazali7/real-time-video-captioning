import torch
from transformers import PreTrainedTokenizer


def encode_caption(caption: str, tokenizer: PreTrainedTokenizer, max_text_len: int = 40) -> torch.Tensor:
    """
    Encode a given string (tokenize + convert to ids) with provided tokenizer.

    Args:
        caption: String to encode
        tokenizer: Tokenizer model
        max_text_len: Integer for longest caption sequence

    Returns:
        Encoded caption ids
    """
    encoding = tokenizer(caption,
                         padding='do_not_pad',
                         truncation=True,
                         add_special_tokens=False,
                         max_length=max_text_len)
    payload = encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    return torch.tensor(input_ids)
