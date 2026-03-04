# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phi4 to Llama bridge model for vLLM.

This module provides a bridge class that allows loading Phi4 format
checkpoints directly into a Llama architecture. It handles:
1. Config patching (adds missing Llama-specific fields like head_dim)
2. Weight transformation (qkv_proj split, gate_up_proj split)
3. Avoids double-loading by skipping init_vllm_registered_model's auto-load

The phi4-siglip checkpoint uses fused weights (qkv_proj, gate_up_proj) while
Llama expects split weights (q_proj, k_proj, v_proj, gate_proj, up_proj).
"""

import logging
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM

root_log = logging.getLogger("vllm")


def _patch_phi4_config_for_llama(hf_config) -> None:
    """Mutate a Phi4-style config to be Llama-compatible.
    
    The Phi4 config (model_type: phi4-siglip) needs these patches to work
    with LlamaForCausalLM:
    - head_dim: calculated as hidden_size // num_attention_heads
    - model_type: changed to "llama" (optional but cleaner)
    
    Args:
        hf_config: The HuggingFace config object to patch (mutated in-place)
    """
    # head_dim is calculated, not stored in Phi4 config
    # Llama config has head_dim: 128, Phi4 has hidden_size: 5120, num_attention_heads: 40
    # 5120 / 40 = 128
    if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
        hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        root_log.info(
            "[Phi4LlamaBridge] Set head_dim = %d (hidden_size=%d / num_attention_heads=%d)",
            hf_config.head_dim,
            hf_config.hidden_size,
            hf_config.num_attention_heads,
        )


class Phi4VisionBridgeLlama(LlamaForCausalLM):
    """LlamaForCausalLM that loads phi4-siglip format weights directly.
    
    This class:
    1. Patches the config in __init__ before LlamaForCausalLM runs
    2. Transforms Phi4->Llama weights during load_weights
    
    Weight transformations:
    - qkv_proj.weight -> q_proj.weight, k_proj.weight, v_proj.weight
    - gate_up_proj.weight -> gate_proj.weight, up_proj.weight
    - All other weights pass through unchanged
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        # Patch config BEFORE parent __init__ runs
        _patch_phi4_config_for_llama(vllm_config.model_config.hf_config)
        root_log.info("[Phi4VisionBridgeLlama] Initialized with patched config")
        
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Transform Phi4 format weights to Llama format, then load.
        
        Filters out vision/projector weights (handled by parent LlamaSiglip).
        
        Args:
            weights: Iterator of (name, tensor) pairs in Phi4 format
            
        Returns:
            Set of loaded parameter names
        """
        root_log.info("[Phi4VisionBridgeLlama] load_weights called - transforming Phi4->Llama")
        
        def transform_iter():
            count = 0
            transformed_count = 0
            for name, tensor in weights:
                # Skip vision and projector weights - parent handles these separately
                if "vision" in name or "mm_projector" in name:
                    continue
                    
                count += 1
                # NOTE: Keep model. prefix - LlamaForCausalLM.load_weights() expects it
                # checkpoint: model.layers.X... -> vLLM Llama expects: model.layers.X...
                
                # Transform Phi4 -> Llama format (qkv_proj split, gate_up_proj split)
                transformed = self._transform_phi4_weight(name, tensor)
                for llama_name, llama_tensor in transformed.items():
                    transformed_count += 1
                    yield llama_name, llama_tensor
            
            root_log.info(
                "[Phi4VisionBridgeLlama] Processed weights: %d LM -> %d transformed",
                count,
                transformed_count,
            )
        
        return super().load_weights(transform_iter())

    def _transform_phi4_weight(
        self, name: str, tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Transform a single Phi4 weight to Llama format.
        
        Args:
            name: Weight name in Phi4 format
            tensor: Weight tensor
            
        Returns:
            Dict mapping Llama weight name(s) to tensor(s)
        """
        # Fast path: pass-through keys (no transformation needed)
        passthrough = (
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
        )
        if any(p in name for p in passthrough):
            return {name: tensor}

        # qkv_proj split: fused QKV -> separate Q, K, V projections
        if "self_attn.qkv_proj.weight" in name:
            n_heads = self.config.num_attention_heads
            n_kv = getattr(self.config, "num_key_value_heads", n_heads)
            head_dim = self.config.hidden_size // n_heads
            expected = (n_heads + 2 * n_kv) * head_dim
            
            if tensor.shape[0] != expected:
                root_log.error(
                    "[Phi4VisionBridgeLlama] Unexpected qkv shape %s (expected first dim %d) for %s",
                    tensor.shape,
                    expected,
                    name,
                )
                raise ValueError(
                    f"Unexpected qkv shape {tensor.shape} "
                    f"(expected first dim {expected}) for {name}"
                )
            
            q_end = n_heads * head_dim
            k_end = q_end + n_kv * head_dim
            q, k, v = tensor[:q_end], tensor[q_end:k_end], tensor[k_end:]
            base = name.replace("qkv_proj.weight", "")
            
            root_log.debug(
                "[Phi4VisionBridgeLlama] Split qkv_proj: %s -> q=%s, k=%s, v=%s",
                name,
                q.shape,
                k.shape,
                v.shape,
            )
            
            return {
                f"{base}q_proj.weight": q,
                f"{base}k_proj.weight": k,
                f"{base}v_proj.weight": v,
            }

        # gate_up_proj split: fused gate+up -> separate gate, up projections
        if "mlp.gate_up_proj.weight" in name:
            gate, up = torch.chunk(tensor, 2, dim=0)
            base = name.replace("gate_up_proj.weight", "")
            
            root_log.debug(
                "[Phi4VisionBridgeLlama] Split gate_up_proj: %s -> gate=%s, up=%s",
                name,
                gate.shape,
                up.shape,
            )
            
            return {
                f"{base}gate_proj.weight": gate,
                f"{base}up_proj.weight": up,
            }

        # Everything else passes through unchanged
        return {name: tensor}
