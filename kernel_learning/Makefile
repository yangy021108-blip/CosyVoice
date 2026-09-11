# ============================================================================
# Kernel learning Makefile —— 一键编译 4 个 CUDA kernel
# ----------------------------------------------------------------------------
# 用法 (需要有 NVIDIA GPU 和 nvcc):
#   make all       # 编译全部
#   make gelu      # 只编译 GELU
#   make run       # 编译并运行全部
#   make clean     # 清理
#
# 如果 GPU 架构不是 sm_80 (A100), 改成你的架构:
#   V100 = sm_70, RTX 3090 = sm_86, RTX 4090 = sm_89
#   查 GPU: nvidia-smi --query-gpu=name,compute_cap --format=csv
#   Colab 里执行: !nvidia-smi -L
# ============================================================================

ARCH ?= sm_80
NVCC := nvcc
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++14

TARGETS := gelu layernorm softmax gemm flash_attention

all: $(TARGETS)

gelu: 01_gelu.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

layernorm: 02_layernorm.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

softmax: 03_softmax.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

gemm: 04_gemm.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

flash_attention: 05_flash_attention.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "=== GELU ===" && ./gelu
	@echo "=== LayerNorm ===" && ./layernorm
	@echo "=== Softmax ===" && ./softmax
	@echo "=== GEMM ===" && ./gemm
	@echo "=== Flash Attention ===" && ./flash_attention

clean:
	rm -f $(TARGETS)

.PHONY: all run clean