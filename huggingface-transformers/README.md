## Requirements

```
pip install transformers>=4.57.1 torch huggingface_hub
```

## Getting Started

### Option A: Download via `huggingface_hub` (Python)

```python
from huggingface_hub import snapshot_download

# Downloads all files (weights + code) to a local cache directory
local_dir = snapshot_download(repo_id="YOUR_HF_REPO_ID")
print(f"Model downloaded to: {local_dir}")

# Then run the sample script from that directory:
#   cd <local_dir> && python sample_inference.py /path/to/image.jpg
```

### Option B: Use `from_pretrained` directly (no manual download needed)

Transformers will automatically download and cache the weights the first time you load the model. The `trust_remote_code=True` flag lets it pull the custom modeling files too:

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

repo_id = "YOUR_HF_REPO_ID"  # e.g. "your-org/edus-test"

processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()
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
