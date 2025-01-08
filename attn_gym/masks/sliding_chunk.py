"""Generates a sliding window attention mask"""

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks
from attn_gym.masks import causal_mask


def generate_sliding_chunk(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_chunk(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= window_size*2) & (kv_idx >= window_size * (q_idx/(window_size)-window_size) )

    sliding_chunk_mask = and_masks(sliding_chunk, causal_mask)
    sliding_chunk_mask.__name__ = f"sliding_chunk_{window_size}"
    return sliding_chunk_mask


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

    sliding_chunk_mask = generate_sliding_chunk(3)
    visualize_attention_scores(
        query, key, mask_mod=sliding_chunk_mask, device=device, name="sliding_chunk_mask"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
