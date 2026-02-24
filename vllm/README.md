# VLLM support

Requirement:
 - vllm version 15.2 (unreleased) or higher

This works only with latest unreleased vllm.

`docker pull vllm/vllm-openai:nightly`

docker run --rm \
    --gpus all \
    --name edus-11 \
    --shm-size=10g \
    --ipc=host \
    --volume ${HOME}/:/workspace/ \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NCCL_P2P_LEVEL=NVL \
    -p 8018:8000 \
    -e AZUREML_MODEL_DIR=/workspace/edus-test \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e VLLM_ARGS=--enforce-eager \
    -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
    --entrypoint  /bin/bash \
    -it vllm/vllm-openai:nightly