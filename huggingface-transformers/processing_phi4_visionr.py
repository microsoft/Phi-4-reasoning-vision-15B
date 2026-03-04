"""
Processor class for Phi4-Siglip.

This module provides:
- Phi4VisionRProcessor: Combined tokenizer and image processor
- Utility functions for image and text processing
"""

from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType

# Constants (duplicated here to avoid circular imports when running scripts directly)
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


# =============================================================================
# Image Utilities
# =============================================================================

def process_images(images: List[Image.Image], image_processor, model_cfg=None):
    """
    Process images for the model.
    
    Args:
        images: List of PIL images
        image_processor: The image processor (Siglip2ImageProcessorNoUpscale for NaFlex)
        model_cfg: Optional model config (unused, kept for API compatibility)
        
    Returns:
        Processed images as BatchFeature (for NaFlex)
    """
    # Check if NaFlex (has max_num_patches attribute)
    is_naflex = hasattr(image_processor, "max_num_patches")
    
    # Process with image processor
    if is_naflex:
        return image_processor(images, return_tensors='pt')
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']


# =============================================================================
# Tokenizer Utilities
# =============================================================================

def tokenizer_image_token(
    prompt: str, 
    tokenizer, 
    image_token_index: int = IMAGE_TOKEN_INDEX, 
    return_tensors: Optional[str] = None
):
    """
    Tokenize a prompt containing <image> tokens.
    
    Replaces <image> with IMAGE_TOKEN_INDEX in the token sequence.
    
    Args:
        prompt: The text prompt with <image> placeholders
        tokenizer: The tokenizer to use
        image_token_index: The index to use for image tokens
        return_tensors: If 'pt', return as PyTorch tensor
        
    Returns:
        List of token ids or tensor
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


# =============================================================================
# Main Processor Class
# =============================================================================

class Phi4VisionRProcessor(ProcessorMixin):
    """
    Processor for Phi4-Siglip that wraps an image processor and tokenizer.
    
    This processor handles:
    - Image preprocessing (via SigLIP or SigLIP2/NaFlex)
    - Text tokenization with image token insertion
    - Conversation formatting
    
    Args:
        image_processor: The image processor (from vision tower)
        tokenizer: The text tokenizer
    """
    
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    image_token = DEFAULT_IMAGE_TOKEN
    
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        """
        Process text and images for the model.
        
        Args:
            text: The text input(s). Can contain <image> tokens.
            images: The image input(s).
            padding: Padding strategy.
            truncation: Whether to truncate.
            max_length: Maximum sequence length.
            return_tensors: Return type for tensors.
            
        Returns:
            BatchFeature with input_ids, attention_mask, and optionally pixel_values.
        """
        # Process images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            image_inputs = process_images(images, self.image_processor)
        else:
            image_inputs = None
        
        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]
            
            # Check if text contains image tokens
            has_images = any(DEFAULT_IMAGE_TOKEN in t for t in text)
            
            if has_images and images is not None:
                # Tokenize with image token handling
                input_ids_list = []
                for t in text:
                    ids = tokenizer_image_token(t, self.tokenizer, return_tensors='pt')
                    input_ids_list.append(ids)
                
                # Pad sequences
                if len(input_ids_list) > 1:
                    max_len = max(len(ids) for ids in input_ids_list)
                    padded_ids = []
                    attention_masks = []
                    pad_token_id = self.tokenizer.pad_token_id or 0
                    
                    for ids in input_ids_list:
                        pad_len = max_len - len(ids)
                        if padding and pad_len > 0:
                            padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
                            attention_masks.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))
                        else:
                            padded_ids.append(ids)
                            attention_masks.append(torch.ones(len(ids)))
                    
                    input_ids = torch.stack(padded_ids)
                    attention_mask = torch.stack(attention_masks).long()
                else:
                    input_ids = input_ids_list[0].unsqueeze(0)
                    attention_mask = torch.ones_like(input_ids)
            else:
                # Standard tokenization
                text_inputs = self.tokenizer(
                    text,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors=return_tensors,
                )
                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs["attention_mask"]
        else:
            input_ids = None
            attention_mask = None

        # Build output
        data = {}
        if input_ids is not None:
            data["input_ids"] = input_ids
            data["attention_mask"] = attention_mask
        
        if image_inputs is not None:
            if isinstance(image_inputs, BatchFeature):
                # NaFlex case - merge all fields
                data.update(image_inputs)
            else:
                data["pixel_values"] = image_inputs
                
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Decode token ids to text. Forwards to tokenizer."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token ids to text. Forwards to tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """Get model input names from tokenizer and image processor."""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = getattr(
            self.image_processor, 
            'model_input_names', 
            ["pixel_values"]
        )
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from a pretrained model path.
        
        This will load the tokenizer and create the appropriate image processor
        based on the model config.
        """
        from transformers import AutoTokenizer, AutoConfig
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Try to load config to determine vision tower type
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            vision_tower_name = getattr(config, 'mm_vision_tower', None)
            vision_config = getattr(config, 'vision_config', None)
            
            if vision_tower_name and 'naflex' in vision_tower_name.lower():
                from .modeling_phi4_visionr import Siglip2ImageProcessorNoUpscale
                # Use embedded vision_config to avoid network calls
                # Infer patch_size from model name if not in config (patch14 vs patch16)
                if vision_config is not None:
                    if 'patch_size' in vision_config:
                        patch_size = vision_config['patch_size']
                    elif 'patch14' in vision_tower_name.lower():
                        patch_size = 14
                    else:
                        patch_size = 16  # default for patch16-naflex
                    image_processor = Siglip2ImageProcessorNoUpscale(
                        patch_size=patch_size,
                        max_num_patches=getattr(config, 'max_num_patches', 3600),
                        min_num_patches=getattr(config, 'min_num_patches', 256),
                    )
                else:
                    image_processor = Siglip2ImageProcessorNoUpscale.from_pretrained(
                        vision_tower_name,
                        max_num_patches=getattr(config, 'max_num_patches', 3600),
                        min_num_patches=getattr(config, 'min_num_patches', 256),
                    )
            elif vision_tower_name:
                from transformers import SiglipImageProcessor
                # Use embedded vision_config to avoid network calls
                if vision_config is not None:
                    image_processor = SiglipImageProcessor(
                        size={"height": vision_config.get('image_size', 384), "width": vision_config.get('image_size', 384)},
                    )
                else:
                    image_processor = SiglipImageProcessor.from_pretrained(vision_tower_name)
            else:
                image_processor = None
        except Exception:
            image_processor = None
            
        return cls(image_processor=image_processor, tokenizer=tokenizer)


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_inputs_for_generation(
    prompt: str,
    images: Optional[List[Image.Image]],
    processor: Phi4VisionRProcessor,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Prepare inputs for model generation.
    
    Args:
        prompt: The user prompt (without conversation formatting)
        images: Optional list of PIL images
        processor: The Phi4VisionRProcessor
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        Dictionary with model inputs
    """
    # Add image token to prompt if images provided
    if images:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    # Use tokenizer's chat_template
    messages = [{"role": "user", "content": prompt}]
    full_prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=full_prompt,
        images=images,
        return_tensors="pt",
    )
    
    # Move to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device=device, dtype=dtype if inputs[key].is_floating_point() else inputs[key].dtype)
    
    return inputs
