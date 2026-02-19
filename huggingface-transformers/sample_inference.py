"""
Sample inference script for Phi4-Siglip.

Usage:
    cd phi4mm
    python sample_inference.py
"""
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "." # change to your model path if not running in the same directory as the model

# get first argument as an image path if not throw an error explaining how to use the script with an image
import sys
with_image_mode = False
if len(sys.argv) > 1:
    with_image_mode = True
    image_path = sys.argv[1]
    print(f"Image path provided: {image_path}")
else:
    print("No image path provided. Running in text-only mode. To run with an image, provide the image path as an argument:\npython sample_inference.py /path/to/image.jpg")

# Load model and processor
print("Loading model...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()

# Get the image token from the processor
image_token = processor.image_token

print(f"Model loaded on {model.device}")

#################################################### text-only ####################################################
print("\n" + "="*60)
print("TEST: Text-only generation")
print("="*60)

messages = [{"role": "user", "content": "What is the answer for 1+1? Explain it."}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f">>> Prompt\n{prompt}")
inputs = processor(prompt, images=None, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(
    **inputs,
    max_new_tokens=4096,
    eos_token_id=processor.tokenizer.eos_token_id,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f'>>> Response\n{response}')

#################################################### single image ####################################################
if not with_image_mode:
    print("\n" + "="*60)
    print("No image provided, skipping multimodal test.")
    print("="*60)
    exit(0)

print("\n" + "="*60)
print("TEST: Single image understanding")
print("="*60)

messages = [{"role": "user", "content": image_token + "\nDescribe this image in detail."}]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

if with_image_mode:
    print(f">>> Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
else:
    image = None

print(f">>> Prompt\n{prompt}")

# Process text and image together using the processor
inputs = processor(text=prompt, images=[image] if image is not None else None, return_tensors="pt").to("cuda:0")

with torch.inference_mode():
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )

generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print(f'>>> Response\n{response}')

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
