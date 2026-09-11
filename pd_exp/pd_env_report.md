# PD 环境调查报告

调查时间：2026-08-25，服务器 `gpu1`。

| 项目 | 实际值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3, compute capability 9.0 |
| Driver | 580.65.06 |
| `nvidia-smi` CUDA capability | 13.0 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Python | 3.10.20 |
| vLLM | 0.11.0，V1 engine |
| Transformers | 4.57.1 |
| NIXL | 0.6.1，独立 venv 安装 |
| UCX | 1.18.0，含 CUDA/verbs/mlx5 支持 |

## GPU 拓扑与 P2P

`nvidia-smi topo -m` 显示所有 H100 之间均为 `NV18`。GPU0/GPU1、
GPU0/GPU2 都是 NVLink/NVSwitch 路径。PyTorch
`torch.cuda.can_device_access_peer()` 对 0→1、0→2、0→4 都返回 `True`。

NIXL agent 成功初始化 UCX backend，可见 plugin 包括：

```text
GDS, GDS_MT, GPUNETIO, LIBFABRIC, OBJ, POSIX, UCX, UCX_MO
```

## 依赖决策

原 vLLM 环境中 `import nixl` 失败。vLLM 0.11.0 对应的
`requirements/kv_connectors.txt` 要求 `nixl>=0.5.1`。本次选择与 vLLM 0.11
已有使用记录相符的 0.6.1，并通过 `--system-site-packages` 独立 venv
安装，没有修改原环境。wheel SHA256：

```text
24e9e98a72839d762bedb8faca010c5878aa0b2d5624a1590d6a588aab1d223e
```

## 资源限制

调查时 GPU1 有用户 `yangyu` 的现有 CosyVoice API/vLLM 进程，GPU3 有
其他用户进程。本实验不会杀这些进程；首次 PoC 选 GPU0/GPU2
做 P/D，GPU4 做 acoustic，它们同样通过 NV18 相连。
