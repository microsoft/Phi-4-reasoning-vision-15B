# inside `setup.py` file
from setuptools import setup

setup(name='phi_4_rv_vllm_plugin',
    version='0.1',
    packages=['phi_4_rv_vllm_plugin'],
    entry_points={
        'vllm.general_plugins':
        ["register_llama_siglip2_model = phi_4_rv_vllm_plugin.llama_siglip2_plugin:register"]
    })
