from phi_4_rv_vllm_plugin.llama_siglip2 import (
    LlamaSiglipForConditionalGeneration,
)
from vllm.model_executor.models import ModelRegistry

def register():
    if "Phi4ForCausalLMV" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Phi4ForCausalLMV",
            LlamaSiglipForConditionalGeneration
        )