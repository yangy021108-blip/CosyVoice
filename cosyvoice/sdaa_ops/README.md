# CosyVoice SDAA 自定义算子

此目录保存 CosyVoice Flow/DiT 推理使用的 TECO SDAA 扩展及完整构建源码。
当前扩展包含：

- 固定 CFG batch 2 的 Flow FlashAttention；
- 基于 TECO LMK 的门控残差、LayerNorm 和 AdaLN scale/shift 融合算子。

FlashAttention 仅在显式设置 `COSYVOICE_SDAA_FLOW_FLASH_ATTN=1`、使用
FP16/BF16、无梯度、batch size 为 2 且 attention mask 全为 `true` 时启用；
其他情况自动回退到 PyTorch SDPA。

融合归一化仅在显式设置 `COSYVOICE_SDAA_FLOW_FUSED_NORM=1`、使用 SDAA
FP16、无梯度且 batch size 为 2 时启用；其他情况自动执行原始 PyTorch
残差、LayerNorm 和 AdaLN 算子序列。

当前二进制面向 Python 3.10 和 TECO 3.2.0 环境。可在
`yy-cosyvoice-sdaa-vllm` 容器中重新构建：

```bash
cd /workspace/CosyVoice/cosyvoice/sdaa_ops
source /opt/tecoai/setvars.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env_py310
python setup.py build_ext --inplace
```

构建成功后可以删除 `cosyvoice/sdaa_ops/build`，但必须保留当前目录下生成的
`.so` 文件。微基准和端到端验证脚本位于仓库的 `tools` 目录，第四轮结果见
`perf_md/cosyvoice_sdaa_vllm_optimization_round4.md`。
