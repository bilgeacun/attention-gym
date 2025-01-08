"""Generates a sliding window attention mask"""

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks
from attn_gym.masks import causal_mask


def generate_chunk_wise(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def chunk_wise(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= window_size) & (kv_idx >= window_size * (q_idx/(window_size)) )

    chunk_wise_mask = and_masks(chunk_wise, causal_mask)
    chunk_wise_mask.__name__ = f"chunk_wise_{window_size}"
    return chunk_wise_mask


def main(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    chunk_wise_mask = generate_chunk_wise(3)
    visualize_attention_scores(
        query, key, mask_mod=chunk_wise_mask, device=device, name="chunk_wise_mask"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
