# VLLM support

## Requirement:
 - vllm version `16.0` (will not work with earlier versions of vllm)
 - `pip install phi_4_rv_vllm_plugin/` (available in this subdirectory only)
    - run command from directory that contains `phi_4_rv_vllm_plugin` dir

## Steps:

### Option A:
Once you have both requirements installed run the following command:

`vllm serve microsoft/Phi-4-reasoning-vision-15B --served-model-name Phi-4-reasoning-vision-15B`

To skip CUDA graphs from being built run previous command with flag `--enforce-eager`



### Option B:
### Build your own Docker image

To simplify the setup you can also build your own image with the following steps:

Running from current directory `Phi-4-reasoning-vision/vllm`

`docker build -t my-custom-image -f ./Dockerfile .`

and then running

```
docker run --gpus all -p 8000:8000 my-custom-image \
  --model microsoft/Phi-4-reasoning-vision-15B \
  --served-model-name Phi-4-reasoning-vision-15B
```

## Reference inference script

A reference inference script is included in this dir called `vllm_serve_test.py`.

Make sure the port number and served model name match with the setup above.

If you run `python3 vllm_serve_test.py` without any args, text only mode will be ran.

If you run `python3 vllm_serve_test.py ./my_dir/my_image.jpg` the image will be used for inference.