# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias
from dataclasses import dataclass

import torch
from torch import nn
from transformers import BatchFeature, ProcessorMixin

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalInputs,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.registry import ModelRegistry

import os
from transformers import SiglipVisionConfig
from vllm.model_executor.models.siglip import SiglipVisionModel
from transformers import Siglip2VisionConfig  # type: ignore
from vllm.model_executor.models.lfm2_siglip2 import Siglip2Model  # type: ignore

from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems

import logging

# Sentinel to prove module import
logging.getLogger("vllm").info("[llama_siglip2] Module imported")

# Replace previous logger usage (keep if needed, but add root logger alias)
root_log = logging.getLogger("vllm")

# Register "phi4-siglip" model_type to use LlamaConfig
# This allows vLLM to load phi4-siglip checkpoints without patching config.json
# The issue is that AutoConfig.from_pretrained rejects unknown model_type values
# but vLLM's _CONFIG_REGISTRY is checked BEFORE AutoConfig
try:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from transformers import LlamaConfig
    if "phi4-siglip" not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["phi4-siglip"] = LlamaConfig
        root_log.info("[llama_siglip2] Registered 'phi4-siglip' -> LlamaConfig in vLLM config registry")
except Exception as e:
    root_log.warning("[llama_siglip2] Failed to register 'phi4-siglip' config: %s", e)

# Import and register the Phi4->Llama bridge model
from .phi4_llama_bridge import Phi4VisionBridgeLlama
try:
    ModelRegistry.register_model("Phi4VisionBridgeLlama", Phi4VisionBridgeLlama)
    root_log.info("[llama_siglip2] Registered Phi4VisionBridgeLlama with vLLM ModelRegistry")
except Exception as e:
    root_log.warning("[llama_siglip2] Failed to register Phi4VisionBridgeLlama: %s", e)


# ---------------------------------------------------------------------------
# Centralized environment configuration - read once at module load.
# Eliminates scattered os.environ lookups throughout the codebase.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionEnvConfig:
    """Immutable environment configuration for vision models.
    
    All environment variable lookups are centralized here and read once
    at module import time.
    
    Note: 
    - naflex_enabled is deduced from vision_config.model_type at runtime:
      model_type='siglip_vision_model' -> SigLIP v1 (S2 path)
      model_type='siglip2_vision_model' -> SigLIP2 NAFLEX
    - max_num_patches/min_num_patches come from hf_config for NAFLEX models
    """
    s2_enabled: bool
    s2_scales: tuple[int, ...]
    
    @classmethod
    def from_environment(cls) -> "VisionEnvConfig":
        """Create config from current environment variables."""
        s2_enabled = os.environ.get("LLAMA_SIGLIP_S2_ENABLED", "0") == "1"
        
        # Parse S2 scales
        scales_env = os.environ.get("LLAMA_SIGLIP_S2_SCALES", "384,768,1152")
        try:
            s2_scales = tuple(sorted(int(x) for x in scales_env.split(",") if x.strip()))
        except Exception:
            s2_scales = (384, 768, 1152)
        
        return cls(
            s2_enabled=s2_enabled,
            s2_scales=s2_scales,
        )


# Singleton instance - read environment once at module load
ENV_CONFIG = VisionEnvConfig.from_environment()


# Optional multiscale S2 wrapper for inference-only image embeddings.
# The implementation lives in `s2wrapper.py` and is adapted from training-time
# TuneImageEmbedding code. Controlled by environment variable
#   LLAMA_SIGLIP_S2_ENABLED=1
# and optional scale overrides
#   LLAMA_SIGLIP_S2_SCALES=384,768,1152
try:
    from .s2wrapper import forward as multiscale_forward  # type: ignore
except Exception:
    multiscale_forward = None  # gracefully degrade if file missing

# Optional: import phi4-siglip SigLIP2 NAFLEX image processor for parity with training
try:
    from .siglip2_encoder import (
        Siglip2ImageProcessorNoUpscale,
    )  # type: ignore
except Exception:
    raise RuntimeError("SigLIP2 NAFLEX processor import failed.")


class _ClsPatchFeature:
    """Baseline single-scale feature selector using a hidden layer index.

    This mirrors TuneImageEmbedding.cls_patch_feature but trimmed for inference.
    """
    def __init__(self, select_layer: int = -2):
        self.select_layer = select_layer

    def feature_select(self, image_forward_outs) -> torch.Tensor:
        return image_forward_outs.hidden_states[self.select_layer]

    def _extract(self, vision_model: SiglipVisionModel, pixel_values: torch.Tensor) -> torch.Tensor:
        """Manual forward up to select_layer (mirrors _extract_intermediate_vision_layer logic)."""
        vt = vision_model.vision_model
        encoder = vt.encoder
        post_ln = getattr(vt, "post_layernorm", None)
        # We intentionally ignore any classification head (use_head) here to preserve
        # patch token grid for multimodal embedding. Pooling to a single token would
        # break alignment with placeholder expansion.
        param = next(vision_model.parameters())
        if pixel_values.device != param.device:
            pixel_values = pixel_values.to(param.device)
        if pixel_values.dtype != param.dtype:
            pixel_values = pixel_values.to(param.dtype)
        cfg_img_size = getattr(vt.config, "image_size", None)
        interpolate_flag = cfg_img_size is not None and pixel_values.shape[-1] != cfg_img_size
        embeds = vt.embeddings(pixel_values, interpolate_pos_encoding=interpolate_flag)
        hidden = embeds
        num_layers = len(encoder.layers)
        layer_index = self.select_layer
        if layer_index < 0:
            layer_index = num_layers - 1 + 1 + layer_index
        if layer_index >= num_layers:
            layer_index = num_layers - 1
        for idx, layer in enumerate(encoder.layers):
            hidden, _ = layer(hidden)
            if idx == layer_index:
                break
        if post_ln is not None and layer_index == num_layers - 1:
            hidden = post_ln(hidden)
        return hidden

    def forward(self, vision_model: SiglipVisionModel, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._extract(vision_model, pixel_values)

    def get_hidden_size(self, vision_model: SiglipVisionModel) -> int:
        return vision_model.config.hidden_size


class _S2MultiscaleFeature(_ClsPatchFeature):
    """Multiscale S2 feature extractor.

    Produces concatenated patch token features across multiple image scales.
    Hidden size multiplies by number of scales. Uses chessboard split to keep memory
    bounded for large resized inputs.
    """
    def __init__(self, scales: list[int]):
        super().__init__(select_layer=-2)
        self.s2_scales = sorted(scales)
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

    def _forward_single(self, vision_model: SiglipVisionModel, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._extract(vision_model, pixel_values)

    def forward(self, vision_model: SiglipVisionModel, pixel_values: torch.Tensor) -> torch.Tensor:
        if multiscale_forward is None:
            raise RuntimeError("multiscale_forward unavailable (s2wrapper not imported)")

        def _wrapped(pv: torch.Tensor) -> torch.Tensor:
            feat = self._forward_single(vision_model, pv)  # [B, N, C]
            if feat.dim() != 3:
                raise ValueError(f"Expected 3D token features, got shape {feat.shape}")
            return feat

        out = multiscale_forward(_wrapped,
                                 pixel_values,
                                 img_sizes=self.s2_scales,
                                 max_split_size=self.s2_split_size,
                                 output_shape='bnc')  # [B, N, C*len(scales)]
        return out

    def get_hidden_size(self, vision_model: SiglipVisionModel) -> int:
        return vision_model.config.hidden_size * len(self.s2_scales)



class DummyProcessor(ProcessorMixin):
    """Processor wrapper combining tokenizer and HF SigLIP image processor.

    if no image_processor is available, images are ignored and only text is tokenized.
    """

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.feature_extractor = None
        self.image_processor = image_processor

    # Minimal helpers --------------------------------------------------------------

    @staticmethod
    def _process_images_pure(image_processor, images) -> object | None:
        """Pure image preprocessing: returns BatchFeature (NAFLEX) or pixel_values tensor, or None."""
        if images is None:
            return None
        if isinstance(images, Sequence) and len(images) == 0:
            return None
        if image_processor is None:
            return None
        batch = image_processor.preprocess(images, return_tensors="pt")
        # If processor returns a dict-like with NAFLEX keys, pass it through.
        if isinstance(batch, BatchFeature) or isinstance(batch, Mapping):
            if "pixel_values" in batch:
                pv = batch["pixel_values"]
                # Ensure float32 for downstream; cast in-place
                if isinstance(pv, torch.Tensor) and pv.dtype != torch.float32:
                    batch["pixel_values"] = pv.float()
            return batch
        # Else, assume tensor
        pixel_values = getattr(batch, "pixel_values", None) if hasattr(batch, "pixel_values") else None
        if pixel_values is None:
            raise ValueError("Image processor returned no 'pixel_values'.")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()
        return pixel_values

    def _process_images(self, images):
        """Impure wrapper around the pure preprocessing for logging; fatal on errors."""
        pv_or_batch = self._process_images_pure(self.image_processor, images)
        if pv_or_batch is None:
            root_log.debug("[DummyProcessor] _process_images: no pixel values produced")
        return pv_or_batch

    def __call__(self, text: str | None = None, images=None, **kwargs):
        # vLLM may pass images via different kwarg names - check all possibilities
        # Priority: explicit 'images' param > 'image' (singular) > 'pixel_values' in kwargs
        if images is None:
            images = kwargs.pop("image", None)
        if images is None:
            images = kwargs.pop("images", None)
        if images is None:
            # Check if raw PIL images were passed with a different key
            for candidate_key in ["pixel_values", "raw_images", "vision_input"]:
                if candidate_key in kwargs:
                    candidate = kwargs.pop(candidate_key)
                    # Only use if it looks like image data (PIL or tensor)
                    from PIL import Image
                    if isinstance(candidate, (Image.Image, list, torch.Tensor)):
                        images = candidate
                        break
        
        if images is None:
            root_log.warning("[DummyProcessor] No images found in any parameter! Available kwargs: %s", list(kwargs.keys()))
        
        # Pass kwargs directly to tokenizer; rely on tokenizer for validation.
        if "return_tensors" not in kwargs:
            kwargs["return_tensors"] = "pt"
        text = text or ""

        tok_out = self.tokenizer(text, **kwargs)
        if not isinstance(tok_out, BatchFeature):
            tok_out = BatchFeature(tok_out)

        pixel_values_or_batch = self._process_images(images)
        out_dict = dict(tok_out)
        if pixel_values_or_batch is not None:
            if isinstance(pixel_values_or_batch, (BatchFeature, Mapping)):
                # Merge all keys (NAFLEX path returns dict with mask + shapes)
                for k, v in dict(pixel_values_or_batch).items():  # type: ignore
                    out_dict[k] = v
            else:
                out_dict["pixel_values"] = pixel_values_or_batch
        # Prune None entries (defensive)
        out_dict = {k: v for k, v in out_dict.items() if v is not None}
        images_present = pixel_values_or_batch is not None
        return BatchFeature(out_dict)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


class LlamaSiglipImagePixelInputs(TensorSchema):
    """Image pixel inputs for LlamaSiglip."""

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]


LlamaSiglipImageInputs: TypeAlias = LlamaSiglipImagePixelInputs
IMAGE_PLACEHOLDER = "<image>"

# ---------------------------------------------------------------------------
# Runtime (immutable) vision configuration to centralize environment + model
# derived settings. This removes scattered os.environ lookups and implicit HF
# config mutations from the rest of the code. Build once, then reuse.
# ---------------------------------------------------------------------------
# Removed RuntimeVisionConfig (Option B: dtype preservation makes it redundant)


def _is_naflex_config(hf_config) -> bool:
    """Check if the vision_config indicates NAFLEX (SigLIP2) mode.
    
    Returns True if vision_config.model_type == 'siglip2_vision_model'.
    Returns False if vision_config.model_type == 'siglip_vision_model' (SigLIP v1).
    """
    existing = getattr(hf_config, "vision_config", None)
    if existing is None:
        return False
    
    # Get model_type from dict or config object
    if isinstance(existing, dict):
        model_type = existing.get("model_type", "")
    elif hasattr(existing, "model_type"):
        model_type = getattr(existing, "model_type", "")
    else:
        return False
    
    return model_type == "siglip2_vision_model"


def _get_naflex_patch_limits(hf_config) -> tuple[int, int]:
    """Get (min_num_patches, max_num_patches) from hf_config for NAFLEX models.
    
    These values are stored in the model's config.json at the top level.
    Returns (256, 3600) as defaults if not found.
    """
    min_patches = getattr(hf_config, "min_num_patches", 256)
    max_patches = getattr(hf_config, "max_num_patches", 3600)
    return (min_patches, max_patches)


# ---------------------------------------------------------------------------
# Shared vision config loader to eliminate duplicated logic between
# LlamaSiglipProcessingInfo.get_vision_config and
# LlamaSiglipForConditionalGeneration.__init__.
# Handles: model id resolution, existing config reuse, remote load + optional
# fallback, and hf_config mutation side-effects.
# ---------------------------------------------------------------------------
def _load_or_create_vision_config(hf_config) -> "SiglipVisionConfig":  # type: ignore
    """Load SigLIP v1 vision config from nested hf_config.vision_config.
    
    Expects model_type='siglip_vision_model' in the nested config.
    No network calls - config must be present in model's config.json.
    """
    from transformers import SiglipVisionConfig

    # Check for nested vision_config in hf_config (no network call needed)
    existing = getattr(hf_config, "vision_config", None)
    if existing is not None:
        # If it's already a SiglipVisionConfig, return it
        if isinstance(existing, SiglipVisionConfig):
            return existing
        # If it's a dict, convert to SiglipVisionConfig
        if isinstance(existing, dict):
            cfg = SiglipVisionConfig(**existing)
            hf_config.vision_config = cfg
            return cfg
        # If it's some other config object, try to extract its dict
        if hasattr(existing, "to_dict"):
            cfg = SiglipVisionConfig(**existing.to_dict())
            hf_config.vision_config = cfg
            return cfg

    raise RuntimeError(
        "No vision_config found in hf_config. Please add vision_config to your model's config.json "
        "with keys: hidden_size, image_size, intermediate_size, model_type, num_attention_heads, num_hidden_layers, patch_size"
    )


def _load_or_create_vision2_config(hf_config) -> "Siglip2VisionConfig":  # type: ignore
    from transformers import Siglip2VisionConfig as _Siglip2VisionConfig  # type: ignore

    # Check for nested vision_config in hf_config (no network call needed)
    existing = getattr(hf_config, "vision_config", None)
    if existing is not None:
        # If it's already a Siglip2VisionConfig, return it
        if isinstance(existing, _Siglip2VisionConfig):
            return existing
        # If it's a dict, convert to Siglip2VisionConfig
        if isinstance(existing, dict):
            cfg = _Siglip2VisionConfig(**existing)
            hf_config.vision_config = cfg
            return cfg
        # If it's some other config object, try to extract its dict
        if hasattr(existing, "to_dict"):
            cfg = _Siglip2VisionConfig(**existing.to_dict())
            hf_config.vision_config = cfg
            return cfg

    raise RuntimeError(
        "No vision_config found in hf_config. Please add vision_config to your model's config.json "
        "with keys: hidden_size, intermediate_size, model_type, num_attention_heads, num_hidden_layers"
    )



class LlamaSiglipProcessingInfo(BaseProcessingInfo):
    _tokenizer_cached: Any | None = None
    _transform_cached: Any | None = None
    _vision_config_cached: Any | None = None  # <-- added cache
    _naflex_enabled_cached: bool | None = None  # <-- cached NAFLEX detection

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def is_naflex_enabled(self) -> bool:
        """Check if NAFLEX mode is enabled based on vision_config.model_type."""
        if self._naflex_enabled_cached is not None:
            return self._naflex_enabled_cached
        hf_config = self.ctx.model_config.hf_config
        self._naflex_enabled_cached = _is_naflex_config(hf_config)
        return self._naflex_enabled_cached

    def get_naflex_patch_limits(self) -> tuple[int, int]:
        """Get (min_num_patches, max_num_patches) from hf_config."""
        hf_config = self.ctx.model_config.hf_config
        return _get_naflex_patch_limits(hf_config)


    def get_tokenizer(self):
        from transformers import AutoTokenizer

        if self._tokenizer_cached is not None:
            return self._tokenizer_cached
        model_id = getattr(self.ctx.model_config, "model", None)
        if not model_id:
            raise ValueError("Model id missing for tokenizer init.")
        tok = AutoTokenizer.from_pretrained(model_id)
        # Ensure <image> single-piece
        encoded = tok.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
        if len(encoded) != 1:
            if IMAGE_PLACEHOLDER not in tok.get_vocab():
                tok.add_special_tokens(
                    {"additional_special_tokens": [IMAGE_PLACEHOLDER]}
                )
            encoded = tok.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError("Failed to coerce <image> token into single piece.")
        self._tokenizer_cached = tok
        return tok

    def get_vision_config(self):
        if self._vision_config_cached is not None:
            return self._vision_config_cached
        hf_config = self.ctx.model_config.hf_config
        cfg = _load_or_create_vision_config(hf_config)
        self._vision_config_cached = cfg
        return cfg


    def get_num_image_tokens(self, width: int, height: int) -> int:
        """Number of patch tokens (spatial stride removed; always full patch grid)."""
        cfg = self.get_vision_config()
        patch = getattr(cfg, "patch_size", 14)
        grid_h = height // patch
        grid_w = width // patch
        return grid_h * grid_w

    def get_image_token_id(self) -> int:
        tok = self.get_tokenizer()
        ids = tok.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
        assert len(ids) == 1
        return ids[0]

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Return the maximum number of tokens per item for each modality.
        
        For NAFLEX, this is the max_num_patches from hf_config.
        This ensures the encoder compute budget is sufficient for the largest images.
        """
        if self.is_naflex_enabled():
            # NAFLEX uses variable-length patches up to max_num_patches
            _, max_tokens = self.get_naflex_patch_limits()
            return {"image": max_tokens}
        else:
            # For non-NAFLEX, use the standard patch grid calculation
            vcfg = self.get_vision_config()
            img_size = getattr(vcfg, "image_size", 384)
            patch_size = getattr(vcfg, "patch_size", 14)
            max_tokens = (img_size // patch_size) ** 2
            return {"image": max_tokens}

    def get_hf_processor(self, **kwargs) -> ProcessorMixin:
        """Return a composite processor combining tokenizer + SigLIP image processor (if available).

        Attempts Fast variant first, then standard. Falls back to torchvision transform only if both fail.
        """
        tokenizer = self.get_tokenizer()
        # Ensure vision config is loaded
        self.get_vision_config()

        image_processor = None
        # Prefer phi4-siglip SigLIP2 NAFLEX processor when NAFLEX detected from config
        naflex_enabled = self.is_naflex_enabled()
        try:
            if naflex_enabled:
                if Siglip2ImageProcessorNoUpscale is not None:
                    # Instantiate directly - no from_pretrained needed!
                    # All defaults in Siglip2ImageProcessorNoUpscale match Google's config:
                    #   do_resize=True, resample=BILINEAR, do_rescale=True,
                    #   rescale_factor=1/255, do_normalize=True,
                    #   image_mean/std=[0.5,0.5,0.5], patch_size=16
                    # max/min patches come from hf_config
                    min_patches, max_patches = self.get_naflex_patch_limits()
                    image_processor = Siglip2ImageProcessorNoUpscale(
                        max_num_patches=max_patches,
                        min_num_patches=min_patches,
                    )
                else:
                    root_log.warning("[LlamaSiglipProcessingInfo] NAFLEX enabled but phi4-siglip SigLIP2 processor unavailable; falling back to HF processors.")
        except Exception as e:
            root_log.warning("[LlamaSiglipProcessingInfo] Failed to create phi4-siglip SigLIP2 processor: %s", e)
        # Try fast processor
        if image_processor is None:
            # Try standard SigLIP v1 processor - instantiate directly from vision_config
            try:
                from transformers import SiglipImageProcessor
                vcfg = self.get_vision_config()
                # SiglipImageProcessor needs: image_size, do_resize, do_normalize, etc.
                # Extract from vision_config or use defaults matching Google's preprocessor_config.json
                image_processor = SiglipImageProcessor(
                    size={"height": getattr(vcfg, "image_size", 384), "width": getattr(vcfg, "image_size", 384)},
                    do_resize=True,
                    do_rescale=True,
                    rescale_factor=1/255,
                    do_normalize=True,
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                )
            except Exception as e_std:
                root_log.warning(
                    "[LlamaSiglipProcessingInfo] SiglipImageProcessor creation failed (%s); falling back to torchvision transform only.",
                    e_std,
                )
        # If we have an HF image_processor, we do not need the torchvision transform for that path.
        # Provide transform only as fallback when image_processor is None.
        proc = DummyProcessor(
            tokenizer,
            image_processor=image_processor,
        )
        # --- S2 multiscale base resize (inference only) ---
        try:
            if ENV_CONFIG.s2_enabled and image_processor is not None:
                scales = list(ENV_CONFIG.s2_scales)
                largest = scales[-1]
                resized = False
                if not resized:
                    image_processor.size['height'] = largest
                    image_processor.size['width'] = largest
                    resized = True
        except Exception as e:
            root_log.warning("[S2] Unexpected failure during processor resize: %s", e)
        # Avoid triggering ProcessorMixin.__repr__ (which JSON dumps non-serializable objects)
        return proc


class LlamaSiglipDummyInputsBuilder(BaseDummyInputsBuilder[LlamaSiglipProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n_img = mm_counts.get("image", 0)
        return (" ".join([IMAGE_PLACEHOLDER] * n_img) + " Profiling").strip()

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None,
                          mm_processor_kwargs=None, **kwargs):
        from PIL import Image

        vcfg = self.info.get_vision_config()
        img_size = getattr(vcfg, "image_size", 224)
        imgs = [
            Image.new("RGB", (img_size, img_size), color=255)
            for _ in range(mm_counts.get("image", 0))
        ]
        return {"image": imgs} if imgs else {}


class LlamaSiglipMultiModalProcessor(
    BaseMultiModalProcessor[LlamaSiglipProcessingInfo]
):

    def _log_hf_inputs_snapshot(self, hf_inputs: BatchFeature):
        try:
            snapshot = {}
            for k, v in hf_inputs.items():
                if isinstance(v, torch.Tensor):
                    snapshot[k] = {
                        "type": "Tensor",
                        "shape": tuple(v.shape),
                        "dtype": str(v.dtype),
                    }
                elif isinstance(v, list):
                    snapshot[k] = {
                        "type": "list",
                        "len": len(v),
                        "elem_types": list({type(x).__name__ for x in v}),
                    }
                else:
                    snapshot[k] = {"type": type(v).__name__}
            root_log.debug(
                "[LlamaSiglipMultiModalProcessor] HF inputs snapshot: %s", snapshot
            )
        except Exception as e:
            root_log.warning(
                "[LlamaSiglipMultiModalProcessor] Failed to build HF inputs snapshot: %s",
                e,
            )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Override to log what data is being passed to the HF processor."""
        # Call base implementation
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _cached_apply_hf_processor(
        self,
        prompt,
        mm_data_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        *,
        mm_uuids = None,
    ):
        """Override to BYPASS caching and always process images directly."""
        # BYPASS cache: Call _apply_hf_processor directly instead of going through cache logic
        # This ensures images always reach the HF processor
        return self._apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        """
        Override to guarantee that when the original prompt is text AND we have
        multimodal items, we always call the text+mm path (so image tensors
        reach the HF processor even during the caching 'missing items' phase
        where enable_hf_prompt_update=False in the base implementation).
        """
        # If prompt already token IDs, defer to base behavior for tokens-only case.
        if not isinstance(prompt, str):
            return super()._apply_hf_processor_main(
                prompt,
                mm_items,
                hf_processor_mm_kwargs,
                tokenization_kwargs,
                enable_hf_prompt_update=enable_hf_prompt_update,
            )

        has_mm = mm_items.get_all_counts() != {}
        if has_mm:
            # Call the text+mm variant unconditionally
            prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )
            # Decide if HF updates were logically applied: only true if caller allowed it AND
            # HF processor is considered to apply updates for these items.
            is_update_applied = (
                enable_hf_prompt_update
                and self._hf_processor_applies_updates(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )
            )
            return prompt_ids, processed_data, is_update_applied
        else:
            return super()._apply_hf_processor_main(
                prompt,
                mm_items,
                hf_processor_mm_kwargs,
                tokenization_kwargs,
                enable_hf_prompt_update=enable_hf_prompt_update,
            )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def apply(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        # Instrumentation: track token lengths before normalization,
        # after normalization, and after multimodal expansion.
        tok = self.info.get_tokenizer()
        image_token_id = self.info.get_image_token_id()

        orig_prompt_str: str | None = None
        orig_token_ids: list[int] | None = None
        norm_token_ids: list[int] | None = None
        num_images: int = 0

        # Log incoming mm_items
        counts = mm_items.get_all_counts()

        if isinstance(prompt, str):
            num_images = counts.get("image", 0)
            orig_prompt_str = prompt
            # Validate placeholders (no mutation)
            placeholder_count = orig_prompt_str.count(IMAGE_PLACEHOLDER)
            if num_images > 0 and placeholder_count != num_images:
                raise ValueError(
                    f"Expected {num_images} <image> placeholders, found {placeholder_count}. Update the prompt to match the image count."
                )
            orig_token_ids = tok.encode(orig_prompt_str, add_special_tokens=False)
            norm_token_ids = orig_token_ids  # no normalization step
        else:
            # If already token IDs, we just record the starting length.
            orig_token_ids = list(prompt)

        # Delegate to base processing (this performs HF processing + placeholder expansion).
        mm_inputs = super().apply(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        final_ids = mm_inputs["prompt_token_ids"]
        final_len = len(final_ids)

        # Count occurrences of the image token id post-expansion.
        image_token_occurrences = sum(1 for tid in final_ids if tid == image_token_id)
        per_image_expansion = (
            (image_token_occurrences / num_images) if num_images > 0 else 0
        )

        return mm_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Heuristic signals about vision presence.
        has_pixel = "pixel_values" in hf_inputs
        has_embeds = "image_embeds" in hf_inputs
        has_raw_images = "images" in hf_inputs  # possible upstream placeholder
        # Also check for NAFLEX keys
        has_naflex_mask = "pixel_attention_mask" in hf_inputs
        has_naflex_shapes = "spatial_shapes" in hf_inputs

        text_only = not (has_pixel or has_embeds or has_raw_images)

        if text_only:
            # Pure text batch – no multimodal fields required.
            root_log.debug(
                "[LlamaSiglipMultiModalProcessor] Treating batch as text-only (no vision fields)."
            )
            self._log_hf_inputs_snapshot(hf_inputs)
            return {}

        # Vision was indicated (raw images or expectation) but nothing processed.
        if not (has_pixel or has_embeds):
            msg = (
                "[LlamaSiglipMultiModalProcessor] Vision inputs indicated but no 'pixel_values' or "
                "'image_embeds' produced (keys=" + str(list(hf_inputs.keys())) + "). Fatal error."
            )
            raise RuntimeError(msg)

        self._log_hf_inputs_snapshot(hf_inputs)

        fields: dict[str, MultiModalFieldConfig] = {}

        if has_pixel:
            pv = hf_inputs["pixel_values"]
            fields["pixel_values"] = MultiModalFieldConfig.batched("image")

            # NAFLEX extras
            if "pixel_attention_mask" in hf_inputs:
                fields["pixel_attention_mask"] = MultiModalFieldConfig.batched("image")
            if "spatial_shapes" in hf_inputs:
                fields["spatial_shapes"] = MultiModalFieldConfig.batched("image")

        if has_embeds:
            emb = hf_inputs["image_embeds"]
            fields["image_embeds"] = MultiModalFieldConfig.batched("image")

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        counts = mm_items.get_all_counts()
        if counts.get("image", 0) == 0:
            return []

        images = mm_items.get_items("image", (ImageProcessorItems, ImageEmbeddingItems))
        image_token_id = self.info.get_image_token_id()
        naflex_enabled = self.info.is_naflex_enabled()

        # NAFLEX-aware placeholder expansion: extract per-image token counts from masks
        lengths: list[int] | None = None

        if naflex_enabled:
            # out_mm_kwargs is MultiModalKwargsItems (UserDict[str, Sequence[MultiModalKwargsItem]])
            # Each MultiModalKwargsItem is UserDict[str, MultiModalFieldElem]
            # MultiModalFieldElem has .data property containing the actual tensor
            try:
                image_items = out_mm_kwargs.get("image")  # Sequence[MultiModalKwargsItem]

                if image_items is not None and len(image_items) > 0:
                    lengths = []
                    for i, item in enumerate(image_items):
                        # item is MultiModalKwargsItem (dict-like)
                        mask_elem = item.get("pixel_attention_mask")
                        if mask_elem is not None:
                            # Get the actual tensor from the field element
                            mask_tensor = getattr(mask_elem, "data", mask_elem)
                            if isinstance(mask_tensor, torch.Tensor):
                                # mask_tensor shape is (Lmax,) for this image - sum to get valid patches
                                n_valid = int(mask_tensor.to(torch.int32).sum().item())
                                lengths.append(n_valid)
                            else:
                                lengths = None
                                break
                        else:
                            lengths = None
                            break
            except Exception as e:
                root_log.warning(
                    "[LlamaSiglipMultiModalProcessor] _get_prompt_updates: Failed to extract NAFLEX masks: %s",
                    e,
                )
                lengths = None

        if lengths is not None and len(lengths) > 0:
            # Capture lengths in closure
            _lengths = lengths

            def replacement(idx: int) -> list[int]:
                n = int(_lengths[idx])
                return [image_token_id] * n

        else:
            vcfg = self.info.get_vision_config()
            patch_size = getattr(vcfg, "patch_size", 16)
            if ENV_CONFIG.s2_enabled:
                ref_size = ENV_CONFIG.s2_scales[0]
            else:
                ref_size = getattr(vcfg, "image_size", 384)

            fixed_tokens = (ref_size // patch_size) * (ref_size // patch_size)

            def replacement(idx: int) -> list[int]:
                if isinstance(images, ImageEmbeddingItems):
                    return [image_token_id] * images.get_feature_size(idx)
                return [image_token_id] * fixed_tokens


        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    LlamaSiglipMultiModalProcessor,
    info=LlamaSiglipProcessingInfo,
    dummy_inputs=LlamaSiglipDummyInputsBuilder,
)
class LlamaSiglipForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """Llama model with SigLIP multimodal infrastructure."""

    supports_multimodal_raw_input_only = True
    merge_by_field_config = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def _configure_vision_dtype_and_device(
        self,
        target_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        """
        (Option B) Align vision tower + projector to language model device, preserving
        their loaded dtype unless an explicit target_dtype is provided. This matches
        prevalent patterns in other multimodal executors (e.g. PaliGemma, AyaVision)
        which rely on vision module weight dtype as authoritative and only cast
        input pixel tensors.

        If target_dtype is None, no dtype casting is attempted. If a cast fails,
        we log and retain the original parameter dtype.
        """
        lm_device = next(self.language_model.parameters()).device

        # Move modules to LM device first.
        if self.vision_tower is not None:
            self.vision_tower.to(lm_device)
        if self.vision2_tower is not None:
            self.vision2_tower.to(lm_device)
        if self.mm_projector is not None:
            self.mm_projector.to(lm_device)
        if target_dtype is not None:
            def _cast_module(module: nn.Module):
                try:
                    module.to(dtype=target_dtype)
                except Exception as e:
                    root_log.warning(
                        "[LlamaSiglip] Vision dtype cast to %s failed for %s: %s; retaining original dtype.",
                        target_dtype,
                        type(module).__name__,
                        e,
                    )
            if self.vision_tower is not None:
                _cast_module(self.vision_tower)
            if self.vision2_tower is not None:
                _cast_module(self.vision2_tower)
            if self.mm_projector is not None:
                _cast_module(self.mm_projector)

    def _extract_intermediate_vision_layer(
        self,
        pixel_values: torch.Tensor,
        layer_index: int = -2,
        apply_post_norm: bool = True,
        pool: bool = False,
    ) -> torch.Tensor:
        """
        Run the SigLIP vision tower up to (and including) a given encoder layer
        and return that hidden state (without the projector applied yet).

        Args:
            pixel_values: [B, 3, H, W] float tensor.
            layer_index: Which encoder layer hidden state to return (0-based).
                         layer_index = 0 returns the output AFTER layer 0.
                         Use -1 to return final layer.
            apply_post_norm: If True, apply post_layernorm only when we are
                             effectively returning the last layer output and
                             the vision model defines post_layernorm.
            pool: If True and the tower has a head (attention pooling), apply it
                  (this collapses sequence length). For intermediate features
                  you usually want pool=False to keep patch tokens.
        Returns:
            hidden_states: [B, L, D] tensor (or [B, 1, D] if pooled).
        """
        if self.vision_tower is None:
            raise RuntimeError("Vision tower not initialized.")
        vt: SiglipVisionModel = self.vision_tower  # type: ignore
        vtrans = vt.vision_model
        encoder = vtrans.encoder
        post_ln = getattr(vtrans, "post_layernorm", None)
        use_head = (
            getattr(vtrans, "use_head", False)
            and getattr(vtrans, "head", None) is not None
        )
        with torch.no_grad():
            # Enable positional interpolation automatically if spatial size differs from config.image_size.
            cfg_img_size = getattr(vtrans.config, "image_size", None)
            interpolate_flag = cfg_img_size is not None and pixel_values.shape[-1] != cfg_img_size
            embeds = vtrans.embeddings(pixel_values, interpolate_pos_encoding=interpolate_flag)
            hidden = embeds
            num_layers = len(encoder.layers)
            if layer_index < 0:
                layer_index = num_layers - 1 + 1 + layer_index
            if layer_index >= num_layers:
                layer_index = num_layers - 1
            for idx, layer in enumerate(encoder.layers):
                hidden, _ = layer(hidden)
                if idx == layer_index:
                    break
            if (
                apply_post_norm
                and post_ln is not None
                and layer_index == num_layers - 1
            ):
                hidden = post_ln(hidden)
            if pool and use_head:
                hidden = vtrans.head(hidden)
        return hidden

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError(f"Unsupported modality: {modality}")

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # Language model - use Phi4VisionBridgeLlama which handles:
        # 1. Config patching (head_dim, etc.)
        # 2. Weight transformation (qkv_proj split, gate_up_proj split)
        # This avoids double-loading: bridge skips auto-load, we load in load_weights()
        lm_config = copy.deepcopy(config)
        lm_config.architectures = ["Phi4VisionBridgeLlama"]
        lm_vllm_config = vllm_config.with_hf_config(
            hf_config=lm_config, architectures=["Phi4VisionBridgeLlama"]
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=lm_vllm_config,
            hf_config=lm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Vision config (centralized loader - no network calls).
        # Detect NAFLEX mode from vision_config.model_type
        self.naflex_enabled: bool = _is_naflex_config(config)

        # Load appropriate vision config based on detected mode
        if self.naflex_enabled:
            vcfg = _load_or_create_vision2_config(config)
        else:
            vcfg = _load_or_create_vision_config(config)
        self.vision_config = vcfg

        # Only create SigLIP v1 tower if NAFLEX is disabled
        if not self.naflex_enabled:
            self.vision_tower = SiglipVisionModel(
                vcfg,
                quant_config=None,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
        else:
            self.vision_tower = None  # Will use vision2_tower instead

        # SigLIP2 NAFLEX vision tower (only when NAFLEX detected)
        self.vision2_config = None
        self.vision2_tower: Siglip2Model | None = None
        if self.naflex_enabled:
            self.vision2_config = vcfg  # Already loaded as Siglip2VisionConfig
            # We use select_layers=[-2], so we only need N-1 layers (skip last layer)
            num_layers_override = vcfg.num_hidden_layers - 1
            self.vision2_tower = Siglip2Model(
                vcfg,
                quant_config=None,
                num_hidden_layers_override=num_layers_override,
                require_post_norm=False,  # Not needed for intermediate layer
                prefix=maybe_prefix(prefix, "vision2_tower"),
            )

        # ---------------- S2 (multiscale) configuration -----------------
        self.s2_enabled: bool = ENV_CONFIG.s2_enabled
        self.s2_scales: list[int] = list(ENV_CONFIG.s2_scales)
        if self.s2_enabled and multiscale_forward is None:
            root_log.warning("S2 requested (LLAMA_SIGLIP_S2_ENABLED=1) but s2wrapper import failed; disabling.")
            self.s2_enabled = False

        # Disable S2 when NAFLEX is on (incompatible)
        if self.naflex_enabled and self.s2_enabled:
            self.s2_enabled = False

        # --- Vision -> Language projector (MLP only; linear removed) ---
        # Hidden dim for projector input (inflate if multiscale enabled)
        base_hidden = (self.vision2_config.hidden_size if (self.naflex_enabled and self.vision2_config is not None) else vcfg.hidden_size)
        self.image_dim_out = base_hidden * (len(self.s2_scales) if self.s2_enabled else 1)
        lm_hidden = self.language_model.config.hidden_size
        mlp_depth = max(1, int(getattr(config, "projection_depth", 2)))  # default depth=2
        dim_projection = getattr(config, "n_embd", lm_hidden)
        layers: list[nn.Module] = [nn.Linear(self.image_dim_out, dim_projection)]
        for _ in range(1, mlp_depth):
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
        self.mm_projector = nn.Sequential(*layers)
        self.projection_cls = "mlp"
        self.projection_depth = mlp_depth

        # Option B: Only co-locate modules on LM device; keep original tower dtype.
        self._configure_vision_dtype_and_device(target_dtype=None)

        # Cache device/dtype to avoid next(parameters()) on every forward call
        if self.vision_tower is not None:
            _vt_p = next(self.vision_tower.parameters())
            self._vision_device = _vt_p.device
            self._vision_dtype = _vt_p.dtype
        elif self.vision2_tower is not None:
            _vt_p = next(self.vision2_tower.parameters())
            self._vision_device = _vt_p.device
            self._vision_dtype = _vt_p.dtype
        else:
            self._vision_device = torch.device("cpu")
            self._vision_dtype = torch.float32
        self._lm_device = next(self.language_model.parameters()).device

        # Feature wrappers (single-scale or multiscale) for raw token features pre-projection.
        self._feature_wrapper = (
            _S2MultiscaleFeature(self.s2_scales) if self.s2_enabled else _ClsPatchFeature(select_layer=-2)
        )
        self._s2_num_scales = len(self.s2_scales) if self.s2_enabled else 1

        # Cache image token id.
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                getattr(vllm_config.model_config, "model", None) or ""
            )
            ids = tok.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
            self.image_token_id = ids[0] if len(ids) == 1 else None
        except Exception:
            self.image_token_id = None

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlamaSiglipImageInputs | None:
        # NAFLEX keys
        naf_pv = kwargs.pop("pixel_values", None)
        naf_mask = kwargs.pop("pixel_attention_mask", None)
        naf_shapes = kwargs.pop("spatial_shapes", None)

        if naf_pv is None:
            return None

        # If mask and shapes are present, treat as NAFLEX patchified input
        if naf_mask is not None and naf_shapes is not None:
            if isinstance(naf_pv, list):
                naf_pv = torch.stack(naf_pv, dim=0)
            if not isinstance(naf_pv, torch.Tensor):
                raise ValueError("pixel_values must be a torch.Tensor (NAFLEX)")
            if not isinstance(naf_mask, torch.Tensor) or not isinstance(naf_shapes, torch.Tensor):
                raise ValueError("NAFLEX requires tensor mask and spatial_shapes")
            return {  # type: ignore
                "type": "naflex",
                "pixel_values": naf_pv,
                "pixel_attention_mask": naf_mask,
                "spatial_shapes": naf_shapes,
            }

        # Fallback: raw pixels path
        pixel_values = naf_pv
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("pixel_values must be a torch.Tensor")
        return LlamaSiglipImagePixelInputs(type="pixel_values", pixel_values=pixel_values)

    # Removed ensure_vision_initialized (lazy path dropped).

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        # Check that we have the appropriate vision tower for the input type
        if self.mm_projector is None:
            raise RuntimeError("mm_projector unexpectedly missing after eager init.")

        # Branch on NAFLEX vs raw pixels
        if isinstance(image_input, Mapping) and image_input.get("type") == "naflex":
            if not self.naflex_enabled or self.vision2_tower is None:
                raise RuntimeError("NAFLEX inputs provided but SigLIP2 vision tower is not enabled.")

            pv = image_input["pixel_values"]  # [B, Lmax, P*P*C]
            mask = image_input["pixel_attention_mask"]  # [B, Lmax]
            shapes = image_input["spatial_shapes"]  # [B, 2]

            dev = self._vision_device
            dt = self._vision_dtype

            # spatial_shapes must stay on CPU (lfm2_siglip2 asserts CPU)
            shapes_cpu = shapes.to("cpu", dtype=torch.long)

            # Derive lengths from spatial_shapes (h*w per image) — no GPU work needed
            lengths_cpu = (shapes_cpu[:, 0] * shapes_cpu[:, 1]).to(torch.int32)
            num_images = shapes_cpu.shape[0]
            cu = torch.zeros(num_images + 1, dtype=torch.int32, device=dev)
            cu[1:] = torch.cumsum(lengths_cpu.to(dev), dim=0)
            maxlen = lengths_cpu.max().to(device=dev)  # must be a tensor, not int

            # Pack on CPU first: extract only non-padded tokens, then transfer to GPU
            bool_mask = mask.to(torch.bool)
            packed_parts = [pv[i][bool_mask[i]] for i in range(num_images)]
            pv_packed = torch.cat(packed_parts, dim=0).unsqueeze(0)  # (1, total_tokens, D)
            pv_packed = pv_packed.to(device=dev, dtype=dt)

            with torch.no_grad():
                # Use select_layers=[-2] to match phi4-siglip training behavior (second-to-last layer)
                vision_hidden = self.vision2_tower(
                    pixel_values_packed=pv_packed,
                    spatial_shapes=shapes_cpu,
                    cu_seqlens=cu,
                    max_seqlen=maxlen,
                    select_layers=[-2],
                )  # (1, total_tokens, D)

                # Batch all images through projector in one call (MLP is batch-agnostic)
                vh = vision_hidden.squeeze(0)  # (total_tokens, D)
                projected_all = self.mm_projector(vh)  # (total_tokens, D_lm)

            # Split projected output per image and move to LM device
            lm_device = self._lm_device
            out_list = []
            offset = 0
            for i in range(num_images):
                n = int(lengths_cpu[i].item())
                chunk = projected_all[offset:offset + n]
                offset += n
                if chunk.device != lm_device:
                    chunk = chunk.to(lm_device, non_blocking=True)
                out_list.append(chunk)
            return out_list
        else:
            # Non-NAFLEX path: requires SigLIP v1 vision tower
            if self.vision_tower is None:
                raise RuntimeError("Non-NAFLEX pixel inputs provided but SigLIP v1 vision tower is not initialized.")
            pv = image_input["pixel_values"]
            # Align device and dtype to cached vision tower parameters.
            if pv.device != self._vision_device:
                pv = pv.to(self._vision_device, non_blocking=True)
            if pv.dtype != self._vision_dtype:
                pv = pv.to(self._vision_dtype)

            target_layer = -2
            pool_flag = False

            with torch.no_grad():
                if self.s2_enabled:
                    vision_hidden = self._feature_wrapper.forward(self.vision_tower, pv)
                else:
                    vision_hidden = self._extract_intermediate_vision_layer(
                        pv,
                        layer_index=target_layer,
                        apply_post_norm=True,
                        pool=pool_flag,
                    )
                projected = self.mm_projector(vision_hidden)

            lm_device = self._lm_device
            if projected.device != lm_device:
                projected = projected.to(lm_device, non_blocking=True)
            if not projected.is_contiguous():
                projected = projected.contiguous()
            embeds = [projected[i] for i in range(projected.shape[0])]
            if len(embeds) != pv.shape[0]:
                raise ValueError(
                    f"Mismatch: batch_images={pv.shape[0]} vs multimodal_embeddings={len(embeds)}"
                )
            return embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,  # match vLLM default
    ) -> torch.Tensor:
        """
        Override vLLM's default embed_input_ids to use custom embedding merge logic.
        Supports flattened 1D token streams ([T]) and batched 2D ([B, S]) inputs.
        """
        # Pure text fallback - use language model's embed_input_ids
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return self.language_model.model.embed_tokens(input_ids)

        if is_multimodal is None:
            raise ValueError("is_multimodal mask required for multimodal merge.")

        # Flatten multimodal embeddings to [N_mm, H]
        if isinstance(multimodal_embeddings, list):
            mm_flat = torch.cat(multimodal_embeddings, dim=0)
        else:
            mm_flat = multimodal_embeddings  # already tensor

        Hdim = mm_flat.shape[-1]
        mm_count = is_multimodal.sum().item()

        # 1D scheduled token path (engine passes a flat vector of all prompt tokens)
        if input_ids.ndim == 1:
            # Text token ids for positions where is_multimodal == False
            is_text = ~is_multimodal
            text_ids = input_ids[is_text]  # [N_text]
            
            n_text = text_ids.numel()
            n_mm = is_multimodal.sum().item()

            text_embeds = self.language_model.model.embed_tokens(
                text_ids
            )  # [N_text, H]
            if text_embeds.shape[-1] != Hdim:
                raise ValueError(
                    f"Hidden size mismatch: text={text_embeds.shape[-1]} mm={Hdim}"
                )

            total_tokens = input_ids.shape[0]
            out = torch.empty(
                total_tokens, Hdim, device=text_embeds.device, dtype=text_embeds.dtype
            )

            text_pos = torch.nonzero(is_text, as_tuple=False).flatten()
            mm_pos = torch.nonzero(is_multimodal, as_tuple=False).flatten()

            if mm_pos.numel() != mm_flat.shape[0]:
                raise ValueError(
                    f"Token count mismatch mm_pos={mm_pos.numel()} vs mm_flat={mm_flat.shape[0]}"
                )

            out.index_copy_(0, text_pos, text_embeds)
            out.index_copy_(0, mm_pos, mm_flat)
            
            return out

        # 2D batched path ([B, S])
        if input_ids.ndim != 2:
            raise ValueError(
                f"Unsupported input_ids.ndim={input_ids.ndim}; expected 1 or 2."
            )

        B, S = input_ids.shape
        is_text = ~is_multimodal
        text_ids = input_ids[is_text]  # flattened over batch
        text_embeds = self.language_model.model.embed_tokens(text_ids)  # [N_text, H]

        if text_embeds.shape[-1] != Hdim:
            raise ValueError(
                f"Hidden size mismatch: text={text_embeds.shape[-1]} mm={Hdim}"
            )

        out = torch.empty(
            B * S, Hdim, device=text_embeds.device, dtype=text_embeds.dtype
        )

        text_pos = torch.nonzero(is_text.view(-1), as_tuple=False).flatten()
        mm_pos = torch.nonzero(is_multimodal.view(-1), as_tuple=False).flatten()

        if mm_pos.numel() != mm_flat.shape[0]:
            raise ValueError(
                f"Token count mismatch mm_pos={mm_pos.numel()} vs mm_flat={mm_flat.shape[0]}"
            )

        out.index_copy_(0, text_pos, text_embeds)
        out.index_copy_(0, mm_pos, mm_flat)

        return out.view(B, S, Hdim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # NOTE: transform_phi3_to_llama_weights has been moved to Phi4VisionBridgeLlama
    # The bridge class now handles all phi4-siglip -> Llama weight transformations

    @staticmethod
    def transform_siglip2_weight_names(
        name: str, tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Transform HuggingFace SigLIP2 weight names to vLLM expected format.
        
        phi4-siglip uses hidden_states[-2] from the encoder, NOT the pooling head output.
        So we skip vision_model.head.* weights (they're unused).
        
        HF uses in_proj_weight/bias for fused QKV, vLLM load_weights expects
        separate q_proj, k_proj, v_proj which it then stacks.
        """
        # Skip pooling head weights - phi4-siglip doesn't use them (uses hidden_states[-2])
        if "vision_model.head." in name:
            return {}
        
        # Encoder self-attention in_proj_weight -> split into q_proj, k_proj, v_proj
        # HF: vision_model.encoder.layers.X.self_attn.in_proj_weight
        # vLLM expects: q_proj.weight, k_proj.weight, v_proj.weight (then stacks into qkv_proj)
        if ".self_attn.in_proj_weight" in name:
            hidden_size = tensor.shape[0] // 3
            q, k, v = tensor.split(hidden_size, dim=0)
            base_name = name.replace(".in_proj_weight", "")
            return {
                f"{base_name}.q_proj.weight": q,
                f"{base_name}.k_proj.weight": k,
                f"{base_name}.v_proj.weight": v,
            }
        
        if ".self_attn.in_proj_bias" in name:
            hidden_size = tensor.shape[0] // 3
            q, k, v = tensor.split(hidden_size, dim=0)
            base_name = name.replace(".in_proj_bias", "")
            return {
                f"{base_name}.q_proj.bias": q,
                f"{base_name}.k_proj.bias": k,
                f"{base_name}.v_proj.bias": v,
            }
        
        # Everything else passes through unchanged:
        # - out_proj.weight, out_proj.bias
        # - layer_norm1/2.weight, layer_norm1/2.bias
        # - mlp.fc1.weight, mlp.fc1.bias
        # - mlp.fc2.weight, mlp.fc2.bias
        # - embeddings.patch_embedding.weight, embeddings.patch_embedding.bias
        # - embeddings.position_embedding.weight
        # - post_layernorm.weight, post_layernorm.bias
        return {name: tensor}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Unified weight loading from vLLM-provided weights iterator.
        
        Single pass through checkpoint to bucket weights, then load in order:
        1. LM weights first (via bridge for Phi4->Llama transformation), then free memory
        2. Vision weights into appropriate tower (v1 or v2/NAFLEX)
        3. Projector weights
        """
        loaded = set()
        mm_projector_state = {}
        vision_state = {}
        lm_weights = []

        # Single pass: bucket weights by component
        weight_count = 0
        for name, tensor in weights:
            weight_count += 1
            if "vision" in name:
                # Strip prefix: model.vision_tower.vision_tower.X -> X
                new_name = name.removeprefix("model.vision_tower.vision_tower.")
                # Transform HF SigLIP2 weight names to vLLM format (split in_proj -> q/k/v)
                transformed = self.transform_siglip2_weight_names(new_name, tensor)
                if not transformed:
                    # Weight was skipped (e.g., pooling head)
                    continue
                for tname, ttensor in transformed.items():
                    vision_state[tname] = ttensor
            elif "mm_projector" in name:
                new_name = name.removeprefix("model.mm_projector.")
                mm_projector_state[new_name] = tensor
            else:
                # LM weights - store for bridge loading
                lm_weights.append((name, tensor))

        # 1. Load LM weights first via bridge (handles Phi4->Llama transformation)
        loaded_lm = self.language_model.load_weights(iter(lm_weights))
        loaded.update({"language_model." + n for n in loaded_lm})
        # Free LM weights memory before loading vision
        del lm_weights

        # 2. Load vision weights into the correct tower based on NAFLEX mode
        if vision_state:
            if self.naflex_enabled and self.vision2_tower is not None:
                vt_loaded = self.vision2_tower.load_weights(
                    (name, tensor) for name, tensor in vision_state.items()
                )
                loaded.update({"vision2_tower." + n for n in vt_loaded})
            elif self.vision_tower is not None:
                vt_loaded = self.vision_tower.load_weights(
                    (name, tensor) for name, tensor in vision_state.items()
                )
                loaded.update({"vision_tower." + n for n in vt_loaded})
        else:
            root_log.warning("[LlamaSiglip] No vision weights found in checkpoint.")

        # Free vision state after loading
        del vision_state

        # 3. Load projector weights
        if mm_projector_state:
            # Convert keys from 'mm_projector.X.weight' -> 'X.weight'
            local_state = {
                k.removeprefix("mm_projector."): v
                for k, v in mm_projector_state.items()
            }
            try:
                missing, unexpected = self.mm_projector.load_state_dict(
                    local_state, strict=False
                )
            except Exception as e:
                root_log.warning("[LlamaSiglip] Failed loading mm_projector: %s", e)
        else:
            root_log.warning("[LlamaSiglip] No projector weights found in checkpoint; using random init.")

        # Free projector state
        del mm_projector_state

        # Record all projector parameter names
        for pname in self.mm_projector.state_dict().keys():
            loaded.add(f"mm_projector.{pname}")

        return loaded
