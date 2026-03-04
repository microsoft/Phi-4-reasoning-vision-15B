#!/usr/bin/env python3
import sys, base64, requests, json, time

SERVED_MODEL_NAME = "Phi-4-reasoning-vision-15B"  # adjust if you used a different name in --served-model-name
PORT = 8000 # VLLM default is 8000, adjust if you used a different port in --port

if len(sys.argv) < 2:
    with_image_mode = False
else:
    with_image_mode = True
    image_path = sys.argv[1]

model_id = SERVED_MODEL_NAME  # adjust if you used --served-model-name
url = f"http://localhost:{PORT}/v1/chat/completions"

if with_image_mode:
    # Encode image as data URI (jpeg assumed; change mime if needed)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(open(image_path, "rb").read()).decode()

    # 1) Multimodal (equivalent to dict with prompt + multi_modal_data['image'])
    payload_image = {
        "model": model_id,
        "messages": [
            {"role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": "What is this?"}
            ]
            }
        ],
        # "images": [data_uri],
        "max_tokens": 4092,
        "temperature": 0.0,
        "stop": ["<|im_end|>"]
    }
else:
    # 2) Text-only (equivalent to plain string prompt)
    payload_text = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "max_tokens": 50,
        "temperature": 0.0,
        "stop": ["<|im_end|>"]
    }

if with_image_mode:
    t0 = time.time()
    r1 = requests.post(url, json=payload_image, timeout=120)
    elapsed = time.time() - t0
    print(f"\n--- Image Prompt Response (HTTP {r1.status_code}, {elapsed:.2f}s) ---")
    print(f"Headers: {dict(r1.headers)}")
    print(json.dumps(r1.json(), indent=2))
else:
    t0 = time.time()
    r2 = requests.post(url, json=payload_text, timeout=60)
    elapsed = time.time() - t0
    print(f"\n--- Text Prompt Response (HTTP {r2.status_code}, {elapsed:.2f}s) ---")
    print(f"Headers: {dict(r2.headers)}")
    print(json.dumps(r2.json(), indent=2))
