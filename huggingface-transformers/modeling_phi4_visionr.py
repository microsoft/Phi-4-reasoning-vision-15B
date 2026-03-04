"""
Minimal self-contained Phi4-Siglip model implementation.

This module provides:
- Phi4VisionR: Configuration class
- Phi4ForCausalLMV: Main vision-language model
- SiglipVisionTower: Vision encoder (standard SigLIP)
- Siglip2VisionTower: Vision encoder with NaFlex (variable token count)
- MLP Projector: Vision-to-language projection
"""

import logging
import os
import re
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Phi3Config,
    Phi3Model,
    Phi3ForCausalLM,
    SiglipVisionModel,
    SiglipVisionConfig,
    SiglipImageProcessor,
    Siglip2VisionModel,
    Siglip2VisionConfig,
    BatchFeature,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import ImagesKwargs
import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips


# =============================================================================
# Constants
# =============================================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


# =============================================================================
# Model Arguments (simplified dataclass for initialization)
# =============================================================================

@dataclass
class ModelArguments:
    """Arguments for model initialization."""
    vision_tower: Optional[str] = None
    vision_tower_path: Optional[str] = None
    mm_projector_type: str = "mlp2x_gelu"
    pretrain_mm_mlp_adapter: Optional[str] = None
    use_s2: bool = False
    s2_scales: str = "384,768,1152"
    hf_cache_dir: Optional[str] = None
    # NaFlex-specific
    min_num_patches: int = 256
    max_num_patches: int = 3600
    # Embedded vision config (to avoid network calls)
    vision_config: Optional[dict] = None


# =============================================================================
# Vision Projector (MLP)
# =============================================================================

def build_vision_projector(config):
    """Build vision-to-language projector based on config."""
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    elif projector_type == 'identity':
        return nn.Identity()

    raise ValueError(f'Unknown projector type: {projector_type}')


# =============================================================================
# Vision Encoders - SigLIP
# =============================================================================

class SiglipVisionTower(nn.Module):
    """Standard SigLIP vision encoder with fixed token count."""
    
    def __init__(self, vision_tower: str, args: ModelArguments = None, delay_load: bool = False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_path = None
        self.select_layer = -2

        self.hf_hub_cache_dir = None
        self.local_files_only = False

        if args and getattr(args, 'hf_cache_dir', None):
            self.hf_hub_cache_dir = args.hf_cache_dir
            self.local_files_only = True
        
        # Load or create vision config once (avoids network calls if embedded config provided)
        vision_config_dict = getattr(args, "vision_config", None) if args else None
        if vision_config_dict is not None:
            self._vision_config = SiglipVisionConfig(**vision_config_dict)
        else:
            self._vision_config = SiglipVisionConfig.from_pretrained(
                self.vision_tower_name,
                local_files_only=self.local_files_only,
                cache_dir=self.hf_hub_cache_dir,
            )
            
        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        # Create image processor
        self.image_processor = SiglipImageProcessor(
            size={"height": self._vision_config.image_size, "width": self._vision_config.image_size},
        )
        self.image_processor.crop_size = self.image_processor.size

        vision_tower_path = self.vision_tower_path if self.vision_tower_path else self.vision_tower_name
        self.vision_tower = SiglipVisionModel.from_pretrained(
            vision_tower_path,
            config=self._vision_config,
            local_files_only=self.local_files_only,
            cache_dir=self.hf_hub_cache_dir,
        )

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        return image_forward_outs.hidden_states[self.select_layer]

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self._vision_config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# =============================================================================
# Vision Encoders - SigLIP2 with NaFlex (variable token count)
# =============================================================================

class Siglip2ImageProcessorKwargsNoUpscale(ImagesKwargs, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int


class Siglip2ImageProcessorNoUpscale(siglip2_ips.Siglip2ImageProcessor):
    """Custom SigLIP2 image processor that doesn't upscale small images."""
    
    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]
    valid_kwargs = Siglip2ImageProcessorKwargsNoUpscale

    def __init__(
        self,
        do_resize: bool = True,
        resample = siglip2_ips.PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: int = 16,
        max_num_patches: int = 256,
        min_num_patches: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches

    @siglip2_ips.filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images,
        resample=None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors=None,
        input_data_format=None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
        max_num_patches: Optional[int] = None,
        min_num_patches: Optional[int] = None,
    ):
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_num_patches = max_num_patches if max_num_patches is not None else self.max_num_patches
        min_num_patches = min_num_patches if min_num_patches is not None else self.min_num_patches

        data_format = siglip2_ips.ChannelDimension.LAST

        try:
            images = self.fetch_images(images)
        except TypeError:
            pass
        images = siglip2_ips.make_flat_list_of_images(images)

        if not siglip2_ips.valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")
        
        siglip2_ips.validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        
        if do_convert_rgb:
            images = [siglip2_ips.convert_to_rgb(image) for image in images]

        images = [siglip2_ips.to_numpy_array(image) for image in images]

        if input_data_format is None:
            input_data_format = siglip2_ips.infer_channel_dimension_format(images[0])

        pixel_masks = []
        pixel_values = []
        spatial_shapes = []

        for image in images:
            image = siglip2_ips.to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            num_patches = max((image.shape[1] // patch_size) * (image.shape[0] // patch_size), 1)
            
            # Resize only if image is too large/small
            if num_patches < min_num_patches:
                height, width = siglip2_ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=min_num_patches,
                )
            elif num_patches > max_num_patches:
                height, width = siglip2_ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=max_num_patches,
                )
            else:
                height, width = siglip2_ips.get_image_size_for_max_num_patches(
                    image_height=image.shape[0],
                    image_width=image.shape[1],
                    patch_size=patch_size,
                    max_num_patches=num_patches,
                )
            
            image = siglip2_ips.resize(image=image, size=(height, width), resample=resample, input_data_format=data_format)

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=data_format)

            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=data_format)

            patches = siglip2_ips.convert_image_to_patches(image, patch_size)
            patches, mask = siglip2_ips.pad_along_first_dim(patches, max_num_patches)
            num_patches_height = image.shape[0] // patch_size
            num_patches_width = image.shape[1] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)
            pixel_masks.append(mask)

        return siglip2_ips.BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_masks,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )


class Siglip2VisionTower(nn.Module):
    """SigLIP2 vision encoder with NaFlex (variable token count per image)."""
    
    def __init__(self, vision_tower: str, args: ModelArguments = None, delay_load: bool = False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_path = None
        self.select_layer = -2

        self.hf_hub_cache_dir = None
        self.local_files_only = False

        self.min_num_patches = getattr(args, "min_num_patches", 256) if args else 256
        self.max_num_patches = getattr(args, "max_num_patches", 3600) if args else 3600

        if args and getattr(args, 'hf_cache_dir', None):
            self.hf_hub_cache_dir = args.hf_cache_dir
            self.local_files_only = True
        
        # Load or create vision config once (avoids network calls if embedded config provided)
        vision_config_dict = getattr(args, "vision_config", None) if args else None
        if vision_config_dict is not None:
            # Infer patch_size from model name if not in config
            if 'patch_size' not in vision_config_dict:
                if 'patch14' in self.vision_tower_name.lower():
                    vision_config_dict['patch_size'] = 14
                else:
                    vision_config_dict['patch_size'] = 16  # default for patch16-naflex
            self._vision_config = Siglip2VisionConfig(**vision_config_dict)
        else:
            self._vision_config = Siglip2VisionConfig.from_pretrained(
                self.vision_tower_name,
                local_files_only=self.local_files_only,
                cache_dir=self.hf_hub_cache_dir,
            )
            
        if not delay_load:
            self.load_model()

    def load_model(self, skip_weights: bool = False):
        """Load the vision tower model.
        
        Args:
            skip_weights: If True, only load the architecture without pretrained weights.
                         Useful when weights will be loaded from a checkpoint later.
        """
        if self.is_loaded:
            return

        # Create image processor
        self.image_processor = Siglip2ImageProcessorNoUpscale(
            patch_size=self._vision_config.patch_size,
            max_num_patches=self.max_num_patches,
            min_num_patches=self.min_num_patches,
        )

        if skip_weights:
            # Load architecture only, no pretrained weights (will load from checkpoint)
            self.vision_tower = Siglip2VisionModel(self._vision_config)
            logger.info("Vision tower initialized without pretrained weights (will load from checkpoint).")
        else:
            vision_tower_path = self.vision_tower_path if self.vision_tower_path else self.vision_tower_name
            self.vision_tower = Siglip2VisionModel.from_pretrained(
                vision_tower_path,
                config=self._vision_config,
                local_files_only=self.local_files_only,
                cache_dir=self.hf_hub_cache_dir,
            )

        self.vision_tower.config.min_num_patches = self.min_num_patches
        self.vision_tower.config.max_num_patches = self.max_num_patches

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        return image_forward_outs.hidden_states[self.select_layer]

    def forward(self, images):
        if isinstance(images, (dict, BatchFeature)):
            images = {
                "pixel_values": images["pixel_values"].to(device=self.device, dtype=self.dtype),
                "pixel_attention_mask": images["pixel_attention_mask"].to(device=self.device, dtype=self.dtype),
                "spatial_shapes": images["spatial_shapes"].cpu().numpy(),
            }
            images_forward_out = self.vision_tower(**images, output_hidden_states=True)
            image_features = self.feature_select(images_forward_out).to(self.dtype)
            # Remove pad tokens
            image_features = [
                feat[images["pixel_attention_mask"][j].bool()] 
                for j, feat in enumerate(image_features)
            ]

        elif isinstance(images, list):
            image_features = []
            for image in images:
                image = {
                    "pixel_values": image["pixel_values"].to(device=self.device, dtype=self.dtype),
                    "pixel_attention_mask": image["pixel_attention_mask"].to(device=self.device, dtype=self.dtype),
                    "spatial_shapes": image["spatial_shapes"].cpu().numpy(),
                }
                image_forward_out = self.vision_tower(**image, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(self.dtype)
                image_feature = [
                    feat[image["pixel_attention_mask"][j].bool()] 
                    for j, feat in enumerate(image_feature)
                ]
                image_features.append(image_feature)
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self._vision_config

    @property
    def hidden_size(self):
        return self.config.hidden_size


# =============================================================================
# Vision Tower Builder
# =============================================================================

def build_vision_tower(config, delay_load: bool = False):
    """Build the appropriate vision tower based on config."""
    vision_tower = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))
    
    if vision_tower is None:
        return None
    
    # Create a minimal args object from config
    args = ModelArguments(
        vision_tower=vision_tower,
        hf_cache_dir=getattr(config, 'hf_cache_dir', None),
        min_num_patches=getattr(config, 'min_num_patches', 256),
        max_num_patches=getattr(config, 'max_num_patches', 3600),
        vision_config=getattr(config, 'vision_config', None),
    )
    
    if 'siglip' in vision_tower.lower():
        if 'naflex' in vision_tower.lower():
            return Siglip2VisionTower(vision_tower, args=args, delay_load=delay_load)
        else:
            return SiglipVisionTower(vision_tower, args=args, delay_load=delay_load)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}. Only SigLIP variants are supported.')


# =============================================================================
# Configuration
# =============================================================================

class Phi4VisionR(Phi3Config):
    """Configuration for Phi4-Siglip model."""
    model_type = "phi4-siglip"
    
    def __init__(
        self,
        mm_vision_tower: Optional[str] = None,
        mm_projector_type: str = "mlp2x_gelu",
        mm_hidden_size: int = 1152,
        min_num_patches: int = 256,
        max_num_patches: int = 3600,
        vision_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mm_vision_tower = mm_vision_tower
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.vision_config = vision_config


# =============================================================================
# Base Model with Vision Integration
# =============================================================================

class Phi4VisionRModel(Phi3Model):
    """Phi3 model with vision tower and projector."""
    config_class = Phi4VisionR

    def __init__(self, config: Phi4VisionR):
        super().__init__(config)

        if hasattr(config, "mm_vision_tower") and config.mm_vision_tower:
            self.vision_tower = build_vision_tower(config, delay_load=not getattr(config, 'continuous_training', False))
            if getattr(config, 'continuous_training', False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if isinstance(vision_tower, list):
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args: ModelArguments):
        """Initialize vision tower and projector from model arguments."""
        vision_tower_name = model_args.vision_tower

        self.config.mm_vision_tower = vision_tower_name

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            if model_args.vision_tower_path:
                vision_tower.vision_tower_path = model_args.vision_tower_path
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        # Ensure projector is trainable
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        # Load pretrained projector weights if provided
        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


# =============================================================================
# Causal LM with Multimodal Support
# =============================================================================

class Phi4ForCausalLMV(Phi3ForCausalLM):
    """Phi4-Siglip model for causal language modeling with vision support."""
    config_class = Phi4VisionR
    
    # Tell transformers to not warn about vision tower weights - we load them separately
    _keys_to_ignore_on_load_unexpected = [r"model\.vision_tower\.vision_tower\..*"]

    def __init__(self, config: Phi4VisionR):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = Phi4VisionRModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """Encode images through vision tower and projector."""
        image_features = self.get_model().get_vision_tower()(images)
        
        # Handle dynamic tokens (NaFlex)
        if isinstance(image_features, list) and isinstance(image_features[0], list):
            image_features = [
                [self.get_model().mm_projector(image) for image in batch] 
                for batch in image_features
            ]
        elif isinstance(image_features, list):
            image_features = [self.get_model().mm_projector(image) for image in image_features]
        else:
            image_features = self.get_model().mm_projector(image_features)
        
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        """
        Prepare inputs by replacing image tokens with actual image embeddings.
        
        This is the core multimodal integration logic that:
        1. Encodes images through the vision tower
        2. Finds IMAGE_TOKEN_INDEX positions in input_ids
        3. Replaces those positions with image embeddings
        4. Handles padding and attention masks
        """
        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # Handle KV cache case during generation
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Encode images
        if (isinstance(images, torch.Tensor) and images.ndim == 5) or \
           (isinstance(images, list) and isinstance(images[0], torch.Tensor)):
            images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(images).to(self.device)
        elif isinstance(images, list) and isinstance(images[0], (dict, BatchFeature)):
            # NaFlex case
            image_features = self.encode_images(images)
            image_features = [image.to(self.device) for batch in image_features for image in batch]
        elif isinstance(images, (dict, BatchFeature)):
            image_features = self.encode_images(images)
            image_features = [image.to(self.device) for image in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        # Store original values
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        # Create defaults if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_temp = input_ids

        # Remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # Replace IMAGE_TOKEN_INDEX with 0 for compatibility
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            if num_images == 0:
                # No image tokens - just embed text
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # Find image token positions
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            # Split by image tokens
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []

            # Interleave text and image embeddings
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],), 
                            IGNORE_INDEX, 
                            device=cur_labels.device,
                            dtype=cur_labels.dtype
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate to max length
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences to same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, 
            dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), 
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), 
            dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
            
            if padding_side == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]), 
                        dtype=cur_new_embed.dtype, device=cur_new_embed.device
                    ),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]), 
                        dtype=cur_new_embed.dtype, device=cur_new_embed.device
                    )
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Restore None values if originally None
        new_labels = None if _labels is None else new_labels_padded
        attention_mask = None if _attention_mask is None else attention_mask.to(dtype=_attention_mask.dtype)
        position_ids = None if _position_ids is None else position_ids

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,     
        logits_to_keep: Union[int, torch.Tensor] = 0,                   
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Accept processor output format (pixel_values, pixel_attention_mask, spatial_shapes)
        if images is None and pixel_values is not None:
            images = BatchFeature({
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            })

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep         
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        
        # Also accept processor output format (pixel_values, pixel_attention_mask, spatial_shapes)
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        
        # If processor output format is provided, package as BatchFeature for the model
        if images is None and pixel_values is not None:
            images = BatchFeature({
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            })

        _inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load model from pretrained weights."""
        # Extract dtype before passing to super() since we need it later
        torch_dtype = kwargs.get("torch_dtype", None)
        
        # Check if loading from local checkpoint that contains vision tower weights
        load_vision_from_checkpoint = False
        if os.path.isdir(pretrained_model_name_or_path):
            for file_name in os.listdir(pretrained_model_name_or_path):
                if file_name.endswith("safetensors"):
                    fpath = os.path.join(pretrained_model_name_or_path, file_name)
                    shard_weights = load_file(fpath)
                    if any(k.startswith("model.vision_tower.vision_tower.") for k in shard_weights.keys()):
                        load_vision_from_checkpoint = True
                        logger.info("Detected vision tower weights in checkpoint - will skip downloading from HuggingFace.")
                        break
        
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        vision_tower = model.get_vision_tower()

        # Load vision weights if model is a local path
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                # Skip downloading pretrained weights if we'll load from checkpoint
                vision_tower.load_model(skip_weights=load_vision_from_checkpoint)

            if load_vision_from_checkpoint:
                try:
                    vision_weights = {}
                    for file_name in os.listdir(pretrained_model_name_or_path):
                        if file_name.endswith("safetensors"):
                            fpath = os.path.join(pretrained_model_name_or_path, file_name)
                            shard_weights = load_file(fpath)
                            
                            # Handle weights with prefix "model.vision_tower.vision_tower."
                            # (the nested vision_tower is the actual encoder)
                            prefix_nested = "model.vision_tower.vision_tower."
                            prefix_simple = "model.vision_tower."
                            
                            for k, v in shard_weights.items():
                                if k.startswith(prefix_nested):
                                    # Strip to get "vision_tower.xxx"
                                    new_key = k[len("model.vision_tower."):]
                                    vision_weights[new_key] = v
                                elif k.startswith(prefix_simple) and not k.startswith(prefix_nested):
                                    # Direct vision_tower weights (like image_processor params if saved)
                                    new_key = k[len(prefix_simple):]
                                    vision_weights[new_key] = v

                    if vision_weights:
                        vision_tower.load_state_dict(vision_weights, strict=False)
                        logger.info("Vision tower weights loaded from checkpoint.")
                    else:
                        logger.warning("No vision tower weights found in checkpoint!")
                except Exception as e:
                    logger.warning(
                        "Vision tower weights NOT loaded from checkpoint. "
                        f"Exception: {e}"
                    )

            vision_tower.to(model.device)

        # Sync dtype
        dtype = torch_dtype if torch_dtype is not None else model.dtype
        dtype = model.dtype if dtype == "auto" else dtype
        model.to(dtype)

        # Fix generation config
        if isinstance(model.generation_config.eos_token_id, (list, set)):
            model.generation_config.eos_token_id = model.generation_config.eos_token_id[0]
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return model


# =============================================================================
# Register with AutoConfig/AutoModel
# =============================================================================

AutoConfig.register("phi4-siglip", Phi4VisionR)
AutoModelForCausalLM.register(Phi4VisionR, Phi4ForCausalLMV)
