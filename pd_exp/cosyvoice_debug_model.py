"""CosyVoice model entry point that installs experiment-only vLLM hooks."""

from pd_exp.graph_runtime import install_vllm_runtime_patch

install_vllm_runtime_patch()

from cosyvoice.vllm.cosyvoice2 import (  # noqa: E402
    CosyVoice2ForCausalLM as _CosyVoice2ForCausalLM,
)


class CosyVoice2ForCausalLM(_CosyVoice2ForCausalLM):
    pass
