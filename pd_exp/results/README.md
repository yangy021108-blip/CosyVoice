# 实验产物说明

该目录在 H100 服务器运行脚本时生成，不把大体积 WAV、PyTorch profiler
trace、序列化 prompt tensor 和批量测试结果提交到 Git。

报告引用的服务器目录为：

```text
/SharedData/yangyu/CosyVoice_pd/pd_exp/results/
```

关键产物包括：

```text
baseline_tokens.json
baseline_metrics.json
baseline.wav
pd_run/prompt_embeds.pt
pd_run/pd.wav
pd_c1/pd_metrics.json
pd_c2/pd_metrics.json
pd_c4/pd_metrics.json
pd_c8/pd_metrics.json
pd_c16/pd_metrics.json
pd_leak_1008/
pd_profile_trace/
unified_interference.json
```

严格复现 token 对比时，必须使用同一份 `pd_run/prompt_embeds.pt`，不能只用
相同文本重新生成 prompt。
