# CosyVoice3 vLLM PD 源码调查

## 现有 CosyVoice 路径

1. `CosyVoiceModel.llm_job()` 将 frontend 的 text/prompt tensor 放到
   `self.device`，然后消费 `self.llm.inference()` 的 token generator。
2. `Qwen2LM.inference()` 拼接 prompt text 和目标 text，用 Qwen embedding
   生成 text embedding，再按顺序拼接 `sos + text + task_id +
   prompt_speech`，得到 `lm_input`。
3. `Qwen2LM.inference_wrapper()` 在 vLLM 分支执行
   `prompt_embeds = lm_input.squeeze(0).to(bfloat16)`，然后调用
   `LLMEngine.add_request()` 与 `LLMEngine.step()`。
4. vLLM 返回的 speech token 逐个 yield 给 `llm_job()`。`llm_job()`
   执行连续 silent-token 过滤后写入 `tts_speech_token_dict[uuid]`。
5. 非流式 `tts()` 等 LLM thread 结束，将完整 speech-token tensor
   交给 `token2wav()`。
6. `CosyVoice2Model/CosyVoice3Model.token2wav()` 先调 Flow `inference()`
   生成 mel，再调 HiFT `inference()` 生成 waveform。

## device 和权重

`CosyVoice3Model.load()` 会把 LLM、Flow、HiFT 都放到同一个
`self.device`。`load_vllm()` 另建一个 vLLM engine，加载
`<model_dir>/vllm` 中的 CosyVoice LLM checkpoint，然后删除原 PyTorch LLM
transformer layers，保留 text/speech embedding 和 acoustic 模块。因此单 vLLM
后端有一份 vLLM LLM weights，不再保留完整的原 LLM layers；Flow/HiFT
仍与 vLLM 共享 `CUDA_VISIBLE_DEVICES` 映射后的同一张卡。

## vLLM 0.11.0 NixlConnector 协议

当前安装的 `KVTransferConfig` 真实 schema 包括：

```text
kv_connector, engine_id, kv_buffer_device, kv_buffer_size, kv_role,
kv_rank, kv_parallel_size, kv_ip, kv_port, kv_connector_extra_config,
kv_connector_module_path
```

NIXL 实际 side-channel 地址使用
`VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`。scheduler 从
`SamplingParams.extra_args["kv_transfer_params"]` 取参数。

Producer 请求使用：

```json
{"do_remote_decode": true, "do_remote_prefill": false}
```

它以 `max_tokens=1` 结束为 `FINISHED_LENGTH_CAPPED`后，connector
延迟释放 prompt KV blocks，并在 `RequestOutput.kv_transfer_params` 返回：

```text
remote_block_ids, remote_engine_id, remote_host, remote_port, tp_size
```

Consumer 使用同一个原始 prompt，附带上述 metadata 和
`do_remote_prefill=true`。`get_num_new_matched_tokens()` 把未在本地计算的
prompt token 全部标记为 external，worker 由 NIXL 异步 READ 直接拉取
GPU KV blocks。

### Prefill 的第一个 token 如何处理

结论是附件中的 B：这是让 vLLM 请求以 length-capped 方式结束、
从而 materialize/保留 prompt KV 的辅助 token，应丢弃。vLLM 0.11
官方 proxy 把原始请求改成 `max_tokens=1` 发给 prefiller，只取
`kv_transfer_params`；随后把未追加该 token 的原始请求发给 decoder。
因此最终 speech-token 序列必须完全来自 decoder。

## prompt embeddings 风险点

Nixl scheduler 用 `len(request.prompt_token_ids)` 计算 external token 数，但
vLLM 0.11 的 prompt-embedding input 按设计令该字段为 `None`。同一个
`Request` 已通过 `length_from_prompt_token_ids_or_embeds()` 正确计算
`num_prompt_tokens`。首次实测因此在 connector scheduler 触发
`TypeError`。

PoC 使用独立 `CosyPromptNixlConnector`，只将这一处长度读取改为
`request.num_prompt_tokens`；不修改 site-packages、KV 传输、scheduler 其他
逻辑或生产路径。PoC 会用两个证据验证这一点：

1. decoder `num_cached_tokens` 应与 prompt embedding 序列长度一致；
2. fixed-seed decoder speech tokens 应与 Unified baseline 逐 token 一致。

任一失败都不进入性能优化。

## 实测修正与版本约束

### vLLM V1 的 chunked prefill

当前 vLLM 0.11.0 V1 engine 即使传入 `enable_chunked_prefill=False`，也会
使用该版本已有的 chunked-prefill scheduler。Unified 与 PD 都运行在同一
版本和同一行为下；PoC 没有引入新的 chunked-prefill 策略，也没有为了速度
改变 scheduler。

### NIXL/UCX 控制面与数据面

服务器原环境没有 NIXL。实验在独立的 system-site-packages venv 中安装
`nixl==0.6.1`，没有修改已有 CosyVoice/vLLM 环境。稳定配置为：

```bash
ulimit -n 65536
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo
```

同机 agent 的 active-message/control channel 使用 loopback TCP，KV buffer
通过 CUDA IPC/copy 读取。仅保留 CUDA transport 时缺少 active-message
transport；枚举全部 mlx5 设备时又会超过默认 1024 soft nofile，因此需要
上述配置。

### external cached token 证据

canonical prompt embedding 长度为 131。Decode worker 实测报告
`num_cached_tokens=130` 和 9 个 remote blocks，说明 decoder 使用远端 prompt
KV，只对最后一个 prompt position/后续 token 执行必要计算，而不是重新做完整
Prefill。根据实际 model/KV dtype、层数和 KV head shape 估算，本请求传输
1,769,472 bytes（约 1.69 MiB）；connector 从 transfer submit 到首次 DONE 的
热态延迟约 1.25–1.42 ms。

### canonical 输入必须序列化复用

只固定文本、参考音频和 seed 仍不足以形成严格 baseline。实测 wetext 在不同
网络状态下对 `CosyVoice3` 产生过不同归一化结果；同一参考音频重新编码也出现
过 1 个 prompt speech token 差异。正确比较必须复用同一个序列化
`prompt_embeds` tensor、spans 和 sampling metadata，不能在不同进程中重新跑
frontend 后再声称输入相同。

## CUDA Graph correctness

验证顺序遵循 eager 先行：

1. PD eager：162/162 raw tokens，154/154 filtered tokens，WAV 与 Unified
   baseline 字节一致；
2. PD `FULL_DECODE_ONLY`：多次稳定生成 164 tokens，而 baseline 为 162；
3. 默认 graph 路径也出现同样漂移。

Graph 路径虽然把 TPOT 从约 6.1 ms 降到约 2.05 ms，但违反逐 token Gate 1，
因此不能作为有效优化保留。当前生产路径不增加 PD feature flag。需要先定位
remote KV 的 block/position state 与 CUDA Graph replay 组合问题，再考虑正式
backend 抽象和集成。

## 干扰实验结论

Unified 中，在 Request A decode 第 40 token 时注入 4,192-token Request B，
A 的注入后 TPOT P99 达 18.71 ms、最大 36.34 ms，并从 reference 162 token
改变为 164 token。这证明同卡 Prefill 会造成 Decode tail spike。

PD 侧若在 A decode 中途新建一个 vLLM EngineCore，worker 在两个候选 GPU 上
都卡在权重加载前，未形成有效对照。不能据此声称 PD 改善了 P95/P99。后续应
预先加载长期存活的 Prefill worker，只在注入点释放执行 barrier。完整记录见
`pd_interference_results.md`。

## 生产集成决策

本轮只保留隔离的 `pd_exp/` 研究 PoC，不修改 `Qwen2LM`、API contract 或默认
backend。原因是当前正确的 eager PD 在 C=16 为 14.42 req/s，低于 Unified
`FULL_DECODE_ONLY` 的 36.16 req/s，并占用两份 LLM weights/KV allocation。
在 correctness 和吞吐 Gate 同时满足前，引入 `NativeBackend/VLLMBackend/
PDVLLMBackend` 及生产 feature flag 会增加维护面，却没有可部署收益。
