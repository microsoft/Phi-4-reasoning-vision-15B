# Running locally with HuggingFace Transformers

## Recommended: Use sample python notebook

Use `sample.ipynb` to install requirements and run two different inference calls with the images contained in this directory.


## Requirements

```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 -U

pip install transformers==4.57.1 Pillow accelerate einops
```

[optional] Install flash-attention:

- Reference `sample.ipynb` to obtain information from your system and pick the right artifact from https://github.com/Dao-AILab/flash-attention/releases

`wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`

`pip install --no-dependencies --upgrade flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`


## Getting Started

### Option A: Use `from_pretrained` directly

Transformers will automatically download and cache the weights the first time you load the model. The `trust_remote_code=True` flag lets it pull the custom modeling files too:

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = "cuda:0"

model_path = "microsoft/Phi-4-reasoning-vision-15B"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device,
    attn_implementation="flash_attention_2", # optional: use "sdpa" if you don't have flash_attention installed
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print(f"Model loaded on {model.device}")
```

## Usage Examples

### Text-only

```python
messages = [{"role": "user", "content": "What is the answer for 1+1? Explain it."}]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, images=None, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(
    **inputs,
    max_new_tokens=4096,
    eos_token_id=processor.tokenizer.eos_token_id,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
```

### Image + Text

```python
from PIL import Image

image = Image.open("your_image.jpg").convert("RGB")

# Use the <image> token to indicate where image content goes
messages = [
    {"role": "user", "content": "<image>\nDescribe this image in detail."}
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda:0")

with torch.inference_mode():
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
response = processor.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print(response)
```

## Image Token

The image placeholder token is available directly from the processor:

```python
image_token = processor.image_token  # "<image>"

messages = [
    {"role": "user", "content": f"{image_token}\nWhat do you see in this image?"}
]
```
