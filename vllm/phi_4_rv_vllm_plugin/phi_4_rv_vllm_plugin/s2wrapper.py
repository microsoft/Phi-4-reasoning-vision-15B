# SPDX-License-Identifier: Apache-2.0
"""Multiscale S2 feature extraction utilities for SigLIP vision tower.

This module provides an optional inference-only multiscale path inspired by
training code (TuneImageEmbedding). It supports splitting large resized images
into chessboard sub-tiles processed in a single batch, merging their outputs,
resizing feature maps across scales, and concatenating them along the channel
(or token feature) dimension.

Enabled via environment variable:
  LLAMA_SIGLIP_S2_ENABLED=1                # turn on multiscale fusion
  LLAMA_SIGLIP_S2_SCALES=384,768,1152      # override default scales
  LLAMA_SIGLIP_S2_MAX_SPLIT=384            # optional split size (default: min(scale_list))

The main entry point is `forward` which mirrors the original signature but is
(lightweight) dependency-free apart from torch/einops.

Contract:
  Args:
    model: callable(pixel_values) -> Tensor[B, N, C] where N is number of patch tokens.
    input: Tensor[B, 3, H, W] square image tensor.
    img_sizes: list[int] target square sizes for resizing & splitting.
    max_split_size: int controlling tile size for chessboard split per scale.
  Returns:
    Tensor[B, N_out, C_concat] where C_concat = C * len(img_sizes) and N_out =
    token count at the reference scale (resize_output_to_idx).

Edge cases handled:
  - Non-square input raises.
  - Mixed dtype interpolation: cast to float32 for interpolation then restore.
  - Prefix tokens not used for SigLIP (kept generic in case of future use).

"""
from __future__ import annotations
import math
from typing import Callable, Sequence
import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "forward",
    "split_chessboard",
    "merge_chessboard",
    "batched_forward",
]


def split_chessboard(x: torch.Tensor, num_split: int) -> torch.Tensor:
    """Split a square image batch into num_split**2 tiles concatenated on batch dim."""
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0, "H/W must be divisible by num_split"
    h, w = H // num_split, W // num_split
    return torch.cat([
        x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
        for i in range(num_split)
        for j in range(num_split)
    ], dim=0)


def merge_chessboard(x: torch.Tensor, num_split: int) -> torch.Tensor:
    """Inverse of split_chessboard (merge tiles back)."""
    B, C, H, W = x.shape
    assert B % (num_split ** 2) == 0, "Batch dimension must be a multiple of num_split**2"
    b = B // (num_split ** 2)
    return torch.cat([
        torch.cat([
            x[(i * num_split + j) * b : (i * num_split + j + 1) * b] for j in range(num_split)
        ], dim=-1)
        for i in range(num_split)
    ], dim=-2)


def batched_forward(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
    if batch_size == -1:
        return model(x)
    outs = []
    for chunk in x.split(batch_size):
        outs.append(model(chunk))
    return torch.cat(outs, dim=0)


def forward(
    model: Callable[[torch.Tensor], torch.Tensor],
    input: torch.Tensor,
    *,
    scales: Sequence[float] | None = None,
    img_sizes: Sequence[int] | None = None,
    max_split_size: int | None = None,
    resize_output_to_idx: int = 0,
    num_prefix_token: int = 0,
    output_shape: str = "bnc",
    split_forward: bool = False,
) -> torch.Tensor:
    """Multiscale forward combining resized chessboard splits.

    Args mirror the training implementation but restricted to inference needs.
    """
    assert input.dim() == 4, "Input must be BxCxHxW"
    assert input.shape[2] == input.shape[3], "Only square inputs supported"
    assert output_shape in ("bnc", "bchw"), "Invalid output_shape"
    if output_shape == "bchw":
        assert num_prefix_token == 0, "ConvNet style output can't have prefix tokens"

    b, c, input_size, _ = input.shape
    assert scales is not None or img_sizes is not None, "Provide scales or img_sizes"
    img_sizes = list(img_sizes) if img_sizes is not None else [int(input_size * s) for s in scales]  # type: ignore

    max_split_size = max_split_size or input_size
    num_splits = [math.ceil(sz / max_split_size) for sz in img_sizes]

    # Prepare multiscale inputs
    multiscale_batches = []
    for sz, n_split in zip(img_sizes, num_splits):
        resized = F.interpolate(input.to(torch.float32), size=sz, mode="bicubic").to(input.dtype)
        tiles = split_chessboard(resized, num_split=n_split)
        multiscale_batches.append(tiles)

    # Model inference (single concatenated call preferred for efficiency)
    chunks = [x.size(0) for x in multiscale_batches]
    big_cat = torch.cat(multiscale_batches, dim=0)
    big_out = model(big_cat)  # Expect [sum(chunks), N_tokens, C]
    outs_multiscale = list(big_out.split(chunks, dim=0))

    # Optional prefix token handling
    if num_prefix_token > 0:
        prefix_tokens = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]

    if output_shape == "bnc":
        # Convert to spatial for chessboard merge (assume square token grid)
        outs_multiscale = [
            rearrange(out, "b (h w) c -> b c h w", h=int(out.shape[1] ** 0.5)) for out in outs_multiscale
        ]

    # Merge tiles back per scale
    outs_multiscale = [merge_chessboard(out, n_split) for out, n_split in zip(outs_multiscale, num_splits)]

    # Area resize to reference scale size and concat along channel
    ref_size = outs_multiscale[resize_output_to_idx].shape[-2]
    outs_resized = [
        F.interpolate(out.to(torch.float32), size=ref_size, mode="area").to(out.dtype) for out in outs_multiscale
    ]
    out = torch.cat(outs_resized, dim=1)  # channel concat

    if output_shape == "bnc":
        out = rearrange(out, "b c h w -> b (h w) c")
    if num_prefix_token > 0:
        prefix_merged = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in prefix_tokens]
        out_prefix = torch.cat(prefix_merged, dim=-1)
        out = torch.cat([out_prefix, out], dim=1)
    return out
