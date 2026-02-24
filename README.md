# Phi-4-Reasoning-Vision

[![Microsoft](https://img.shields.io/badge/Microsoft-Project-0078D4?logo=microsoft)](https://aka.ms/msaif/fara)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/microsoft/Fara-7b)
[![Foundry](https://img.shields.io/badge/Azure-Foundry-0089D6)](https://aka.ms/foundry-fara-7b)
[![Paper](https://img.shields.io/badge/Paper-2511.19663-red)](https://arxiv.org/abs/2511.19663)

## Overview

Phi-4-Reasoning-Vision-15B is a broadly capable model that can be used for a wide array of vision-language tasks such as image captioning, asking questions about images, reading documents and receipts, helping with homework, interfering about changes in sequences of images, and much more. Beyond these general capabilities it excels at math and science reasoning and at understanding and grounding elements on computer and mobile screens.

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

Deploy Fara-7B on [Azure Foundry](https://ai.azure.com/explore/models/Fara-7B/version/2/registry/azureml-msr) without needing to download weights or manage GPU infrastructure.

**Setup:**

1. Deploy the Fara-7B model on Azure Foundry and obtain your endpoint URL and API key

Then create a endpoint configuration JSON file (e.g., `azure_foundry_config.json`):

```json
{
    "model": "Fara-7B",
    "base_url": "https://your-endpoint.inference.ml.azure.com/",
    "api_key": "YOUR_API_KEY_HERE"
}
```

Then you can run Fara-7B using this endpoint configuration.

2. Run the Fara agent:

```bash
fara-cli --task "how many pages does wikipedia have" --endpoint_config azure_foundry_config.json [--headful]
```

Note: you can also specify the endpoint config with the args `--base_url [your_base_url] --api_key [your_api_key] --model [your_model_name]` instead of using a config JSON file. 

Note: If you see an error that the `fara-cli` command is not found, then try:

```bash
python -m fara.run_fara --task "what is the weather in new york now"
```

That's it! No GPU or model downloads required.

### Self-hosting with vLLM

If you have access to GPU resources, you can run inference using HuggingFace Transformers library. This requires a GPU machine with sufficient VRAM (e.g., 40GB or more).

Instructions available [here](https://github.com/microsoft/Phi-4-reasoning-vision-15B/tree/main/huggingface-transformers).

### Self-hosting with vLLM

If you have access to GPU resources, you can self-host using vLLM. This requires a GPU machine with sufficient VRAM (e.g., 40GB or more).

Instructions available [here](https://github.com/microsoft/Phi-4-reasoning-vision-15B/tree/main/vllm).

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