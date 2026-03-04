# Phi-4-reasoning-vision-15B

[![Microsoft](https://img.shields.io/badge/Microsoft-Project-0078D4?logo=microsoft)](https://aka.ms/Phi-4-r-v-FoundryLabs)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)
[![Foundry](https://img.shields.io/badge/Azure-Foundry-0089D6)](https://ai.azure.com/catalog/models/Phi-4-Reasoning-Vision-15B)
[![Paper](https://img.shields.io/badge/Paper-2511.19663-red)](https://arxiv.org/abs/<>)

## Overview

Phi-4-reasoning-vision-15B is a broadly capable model that can be used for a wide array of vision-language tasks such as image captioning, asking questions about images, reading documents and receipts, helping with homework, interfering about changes in sequences of images, and much more. Beyond these general capabilities it excels at math and science reasoning and at understanding and grounding elements on computer and mobile screens.

## Benchmark Results

We've tested on the following benchmarks:
-----

| **Benchmark**               | **Link** |
|-----------------------------|-----------|
| AI2D                        | [lmms-lab/ai2d · Datasets at Hugging Face](https://huggingface.co/datasets/lmms-lab/ai2d) |
| HallusionBench              | [lmms-lab/HallusionBench · Datasets at Hugging Face](https://huggingface.co/datasets/lmms-lab/HallusionBench) |
| MathVerse                   | [AI4Math/MathVerse · Datasets at Hugging Face](https://huggingface.co/datasets/AI4Math/MathVerse) |
| MathVision                  | [MathLLMs/MathVision · Datasets at Hugging Face](https://huggingface.co/datasets/MathLLMs/MathVision) |
| MathVista                   | [AI4Math/MathVista · Datasets at Hugging Face](https://huggingface.co/datasets/AI4Math/MathVista) |
| MMMU                        | [MMMU/MMMU · Datasets at Hugging Face](https://huggingface.co/datasets/MMMU/MMMU) |
| MMStar                      | [Lin-Chen/MMStar · Datasets at Hugging Face](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| ScreenSpot v2               | [Voxel51/ScreenSpot-v2 · Datasets at Hugging Face](https://huggingface.co/datasets/Voxel51/ScreenSpot-v2) |
| WeMath                      | [We-Math/We-Math · Datasets at Hugging Face](https://huggingface.co/datasets/We-Math/We-Math) |
| ZEROBench                   | [jonathan-roberts1/zerobench · Datasets at Hugging Face](https://huggingface.co/datasets/jonathan-roberts1/zerobench) |

## Benchmark Results

| **Benchmark**              | **Score** |
|-----------------------------|-----------|
| AI2D_TEST                   | 84.8      |
| HallusionBench              | 64.4      |
| MathVerse_MINI              | 44.9      |
| MathVision_MINI             | 36.2      |
| MathVista_MINI              | 75.2      |
| MMMU_VAL                    | 54.3      |
| MMStar                      | 64.5      |
| ScreenSpot_v2_Desktop       | 87.1      |
| ScreenSpot_v2_Mobile        | 88.6      |
| ScreenSpot_v2_Web           | 88.8      |
| WeMath                      | 50.1      |
| ZEROBench_sub               | 17.7      |

## Hosting the Model

**Recommended:** The easiest way to get started is using Azure Foundry hosting, which requires no GPU hardware or model downloads. Alternatively, you can self-host with vLLM if you have GPU resources available.

### Azure Foundry Hosting (Recommended)

Deploy Phi-4-Reasoning-Vision-15B on [Azure Foundry](https://ai.azure.com/catalog/models/Phi-4-Reasoning-Vision-15B) without needing to download weights or manage GPU infrastructure.

**Setup:**

1. Deploy the model on Azure Foundry and obtain your endpoint URL, API key and deployment name.

Use the following sample script, be sure to replace the following:
- IMAGE_PATH, ENDPOINT_BASE, API_KEY, DEPLOYMENT_NAME
- Optional: content of the payload message



```
import base64
import os
import requests

IMAGE_PATH = "<replace_with_your_image>.jpg"

ENDPOINT_BASE = "<your_base_endpoint_url>"
API_KEY = "<your_api_key_here>"
DEPLOYMENT_NAME = "Phi-4-Reasoning-Vision-15B" # replace here with your deployment name

def main():
    with open(IMAGE_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "Phi-4-Reasoning-Vision-15B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "azureml-model-deployment": DEPLOYMENT_NAME,
    }

    url = f"{ENDPOINT_BASE}/v1/chat/completions"
    print(f"Requesting: {url}")

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    result = resp.json()
    print("\n--- Response ---")
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()

```



That's it! No GPU or model downloads required.

### Self-hosting with Transformers

If you have access to GPU resources, you can run inference using HuggingFace Transformers library. This requires a GPU machine with sufficient VRAM (e.g., 40GB or more).

Instructions available [here](https://github.com/microsoft/Phi-4-reasoning-vision-15B/blob/main/huggingface-transformers/README.md).

### Self-hosting with vLLM

If you have access to GPU resources, you can self-host using vLLM. This requires a GPU machine with sufficient VRAM (e.g., 40GB or more).

Instructions available [here](https://github.com/microsoft/Phi-4-reasoning-vision-15B/blob/main/vllm/README.md).

## Citation

If you use Phi-4-Reasoning-Vision in your research, please use the following BibTeX entry.
```bibtex
@article{phi4vr14b2026,
  title={Phi-4-Vision-Reasoning Technical Report},
  author={Aneja, Jyoti and Harrison, Michael and Joshi, Neel and LaBonte, Tyler and Langford, John and Salinas, Eduardo and Ward, Rachel},
  journal={arXiv:2511.19663},
  year={2026}
}
```