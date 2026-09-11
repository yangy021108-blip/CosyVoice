# CosyVoice3 vLLM Prefill-Decode 实验

该目录是与生产 API 解耦的研究 PoC。它使用真实 CosyVoice3
`prompt_embeds`，让 Prefill/Decode 运行在不同进程和不同 H100 上，KV 由
vLLM 0.11.0 `NixlConnector` 在 GPU memory 间传输。默认 CosyVoice、单
vLLM 和 API 路径均未改变。

## 环境

PD 环境只在现有 vLLM 环境外增加 NIXL：

```bash
/SharedData/yangyu/cosyvoice_vllm_env/cosyvoice/bin/python -m venv --system-site-packages /SharedData/yangyu/cosyvoice_pd_env
/SharedData/yangyu/cosyvoice_pd_env/bin/python -m pip install --no-deps nixl==0.6.1
```

每个实验 shell 先执行：

```bash
ulimit -n 65536
cd /SharedData/yangyu/CosyVoice_pd
export PYTHONPATH=/SharedData/yangyu/CosyVoice_pd:/SharedData/yangyu/CosyVoice/third_party/Matcha-TTS
export TOKENIZERS_PARALLELISM=false
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo
```

loopback TCP 只承担 NIXL active-message 控制面；KV 数据使用 CUDA
IPC/copy。不要去掉 TCP，否则 UCX 无法建立控制通道。

## 1. Unified baseline

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/baseline.py --gpu 0 --matcha-path /SharedData/yangyu/CosyVoice/third_party/Matcha-TTS
```

生成：

```text
pd_exp/results/baseline_tokens.json
pd_exp/results/baseline_metrics.json
pd_exp/results/baseline.wav
pd_exp/results/prompt_payload.pt
```

跨进程比较必须复用同一份 `prompt_payload.pt`。只固定文本不够：实测
wetext 可因网络状态改变文本归一化，参考音频重新编码也出现过 1 个
prompt speech token 漂移。

## 2. 1P1D correctness + audio

GPU0=Prefill、GPU2=Decode、GPU4=Flow/HiFT：

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/coordinator.py --prefill-gpu 0 --decode-gpu 2 --acoustic-gpu 4 --baseline-tokens pd_exp/results/baseline_tokens.json --prompt-payload pd_exp/results/pd_run/prompt_embeds.pt --results-dir pd_exp/results/pd_run --matcha-path /SharedData/yangyu/CosyVoice/third_party/Matcha-TTS
```

只有 raw 和 filtered token 都逐 token 相等才通过 Gate 1。当前 canonical
结果为 162/162 raw、154/154 filtered，PD WAV 与 baseline WAV 字节完全
一致。

## 3. 稳态 KV 和并发

先丢弃第 0 轮首次 NIXL handshake，只统计第 1 轮：

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/coordinator.py --prefill-gpu 0 --decode-gpu 2 --acoustic-gpu 4 --iterations 2 --batch-size 4 --instrument-nixl --skip-acoustic --baseline-tokens pd_exp/results/baseline_tokens.json --prompt-payload pd_exp/results/pd_run/prompt_embeds.pt --results-dir pd_exp/results/pd_c4 --matcha-path /SharedData/yangyu/CosyVoice/third_party/Matcha-TTS
```

Unified 对照：

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/benchmark_pd.py --gpu 0 --prompt-payload pd_exp/results/pd_run/prompt_embeds.pt --baseline-tokens pd_exp/results/baseline_tokens.json --concurrency 4 --iterations 2 --engine-mode standard_optimized --output pd_exp/results/unified_standard_optimized_c4.json
```

`--standard-optimized` 在 PD 上会启用现有 `FULL_DECODE_ONLY` graph，但当前
会从 162 token 漂移为 164 token，因此只保留作失败实验，不能作为有效
PD 性能结果。正确 PD 使用默认 eager。

## 4. Profiler

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/coordinator.py --prefill-gpu 0 --decode-gpu 2 --acoustic-gpu 4 --iterations 2 --batch-size 1 --instrument-nixl --torch-profile --skip-acoustic --baseline-tokens pd_exp/results/baseline_tokens.json --prompt-payload pd_exp/results/pd_run/prompt_embeds.pt --results-dir pd_exp/results/pd_profile_trace --matcha-path /SharedData/yangyu/CosyVoice/third_party/Matcha-TTS
```

输出包括：

```text
pd_exp/results/pd_profile_trace/prefill_profile/*.pt.trace.json.gz
pd_exp/results/pd_profile_trace/decode_profile/*.pt.trace.json.gz
pd_exp/results/pd_profile_trace/decode_trace.jsonl
```

服务器未安装 Nsight Systems，因此没有 `.nsys-rep`；connector JSONL 是
KV prepare/submit/DONE 的独立时间线。代码还提供 `COSY_PD_PREFILL`、
`COSY_PD_DECODE`、`COSY_FLOW`、`COSY_HIFT` NVTX ranges，以及 pull-based
NIXL 的 `COSY_PD_KV_SEND`（KV 已暴露）、`COSY_PD_KV_RECV` 和
`COSY_PD_KV_RECV_DONE` marks。异步 KV 的精确耗时以 JSONL 为准，不能用
可能交错的 host NVTX range 冒充传输时长。

## 5. 清理检查

脚本只终止自己创建的 worker。人工检查时只匹配本实验，不要杀其他用户
或已有 API：

```bash
pgrep -af 'pd_exp/(prefill_worker|decode_worker|coordinator|benchmark_pd)\.py'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

完整数据和结论见根目录 `PD_H100_REPORT.md`；干扰实验的边界与失败记录见
`pd_interference_results.md`。
