# Kernel 学习指南 & 面试准备

> 配套文件：`01_gelu.cu` / `02_layernorm.cu` / `03_softmax.cu` / `04_gemm.cu` / `05_flash_attention.cu` / `reference.py`

---

## 一、怎么编译运行（需要 NVIDIA GPU）

本机没有 GPU，用 **Google Colab（免费）** 或任何有 CUDA 的机器：

```bash
# 单个编译运行
nvcc -O3 -arch=sm_80 01_gelu.cu -o gelu && ./gelu
nvcc -O3 -arch=sm_80 02_layernorm.cu -o ln && ./ln
nvcc -O3 -arch=sm_80 03_softmax.cu -o sm && ./sm
nvcc -O3 -arch=sm_80 04_gemm.cu -o gemm && ./gemm
nvcc -O3 -arch=sm_80 05_flash_attention.cu -o flash && ./flash
```

`-arch=sm_80` 按你的 GPU 改（A100=sm_80, V100=sm_70, 3090=sm_86, 4090=sm_89）。
不确定就用 `nvidia-smi` 查，或 Colab 里直接 `!nvcc --list-gpu-arch`。

**Colab 用法**：新建 notebook → 代码单元格里 `%%writefile 01_gelu.cu` 粘贴代码 → 再开一个单元格 `!nvcc ... && !./gelu`。

---

## 二、学习顺序（强烈建议按这个顺序，别跳）

### 第 0 步：先跑通 `reference.py`（本机就能跑）
看懂 4 个算法的**数学**。这一步不涉及 GPU，是地基。
- 重点：`online_softmax` 的递推、`layernorm` 的 mean/var 公式。

### 第 1 步：GELU（elementwise）
理解 GPU 编程的基本模型：
- **grid / block / thread** 三级层次
- 全局线程 id 怎么算：`blockIdx.x * blockDim.x + threadIdx.x`
- **合并访存**：相邻线程访问相邻内存地址
- 边界处理：`if (idx >= N) return`

✅ 检验标准：你能解释为什么 `block=256`、为什么要 `if (idx>=N) return`、float4 为什么更快。

### 第 2 步：LayerNorm（reduction）
理解**线程协作**：
- 为什么需要 `__syncthreads()`
- warp 内 `__shfl_down_sync` 规约
- shared memory 广播规约结果
- 一个 block 处理一行的设计

✅ 检验标准：你能画出"32 个线程如何把 32 个数归约成 1 个和"的树形图。

### 第 3 步：Softmax（online 算法）
这是**最重要**的一步，直接连到 Flash Attention：
- 为什么要减 max（数值稳定）
- **online softmax 的 rescale 递推**（面试必考）
- 理解 `merge_ml` 如何在并行规约中合并 (max, sum) 对

✅ 检验标准：你能手推 online softmax 的递推公式，并说出它和 Flash Attention 的关系。

### 第 4 步：GEMM（tiling）
理解**内存层次与复用**：
- naive 为什么慢（全局内存反复读）
- shared memory tiling 如何复用
- **算术强度**（arithmetic intensity）概念
- 进阶：寄存器分块、双缓冲、Tensor Core

✅ 检验标准：你能解释 tile=32 怎么选、双缓冲解决什么问题。

### 第 5 步：Flash Attention（终极目标）
把前 4 个 kernel 全部串起来：
- **tiled GEMM**（Kernel 04）：计算 Q_block @ K_block^T
- **online softmax**（Kernel 03）：对每个 Q 行维护 running (max, sum)
- **elementwise + reduction**（Kernel 01+02）：rescale 输出累加器

核心思想：
- 不存 N×N 的 S 矩阵，显存从 O(N²) 降到 O(N)
- Q 行加载到寄存器只读一次 global memory，K/V 分块加载到 shared memory
- 每处理一个 K/V 块，用 online softmax 更新 (m, l, O)，遇到更大 max 就 rescale
- 最终 O /= l 得到答案

✅ 检验标准：
- 你能手推 online softmax 的 rescale 公式
- 你能说出"rescale 必须放在内层循环外面"的原因（这是最常见的 bug）
- 你能算出 naive vs Flash 的显存差异（N=4096 时差距 ~2000×）

🎯 这一步完成后，你就能在面试里讲清楚 CosyVoice 里 FlashAttention 的算法了。

---

## 三、面试怎么讲（诚实且有力的版本）

### ❌ 不要这样说
> "我设计了一个 Flash Attention kernel。"
（如果面试官问 kernel 内部 tiling 细节、你答不上来，就露馅了）

### ✅ 推荐这样说（分两种情况）

**情况 A：讲你的 CosyVoice 真实工作（这是你最硬的资产）**
> "我在 CosyVoice 上做 SDAA 加速卡的推理优化，端到端延迟降了 49.8%。
> 其中我写了 C++ PyTorch 扩展，通过 `TORCH_LIBRARY` 注册自定义算子，
> 把 TECO 底层库的 Flash Attention 和 fused norm 接入 PyTorch。
> 我自己实现的是**算子适配层**：handle 管理、workspace 分配、tensor 描述符转换、
> 以及 Python 端的守卫条件和回退逻辑。真正的 GPU kernel 在 TECO 闭源库里。"

这样说**完全真实**，而且展示了你懂"算子接入"这一层的工程能力。面试官如果问 kernel 内部，你可以坦然说"kernel 实现在厂商库里，我负责的是适配和集成"——这是诚实且专业的回答。

**情况 B：讲你自己写的这些学习 kernel**
> "为了深入理解这些优化，我自己用 CUDA 实现了 GELU、LayerNorm、online Softmax
> 和 tiled GEMM。比如 online softmax，我用一遍扫描维护 (max, sum) 状态，
> 遇到更大的 max 就用 exp(m_old - m_new) 去 rescale 已有的 sum——
> 这正是 Flash Attention 分块计算的数学基础。"

这样说**也是真实的**（你确实写了、理解了），而且能扛住追问，因为你真的懂每一行。

### 关键原则
1. **只声称你真正做了 + 能讲清的**。能讲清 = 能回答 3 层追问。
2. **区分"实现"和"设计"**：你实现/复现了 kernel 是事实；说"我从零设计了 Flash Attention"就不是。用词精确，面试官反而更信任你。
3. **把学习和真实工作串起来**：你的 kernel 学习不是孤立的，它服务于解释你的 CosyVoice 优化。这条线非常加分。

---

## 四、面试高频追问清单（背熟 + 真懂）

| 问题 | 考点 | 你要能答 |
|------|------|---------|
| 为什么 softmax 要减 max？ | 数值稳定 | exp 溢出 |
| online softmax 递推公式？ | Flash Attention 基础 | rescale 机制 |
| Flash Attention 为什么不存 N×N 矩阵？ | 内存 | 分块 + online softmax |
| GEMM tiling 为什么快？ | 内存复用 | shared mem + 算术强度 |
| 什么是算术强度？怎么判断瓶颈？ | roofline | FLOPs/字节 |
| warp 是什么？为什么规约用 shuffle？ | SIMT | 32 线程一组 |
| 什么时候用 shared memory？ | 内存层次 | 复用 + 低延迟 |
| elementwise kernel 瓶颈是什么？ | memory-bound | 带宽 |
| 你的 fused norm 为什么能加速？ | 减少访存 | 多 op 合一 |

---

## 五、下一步进阶（学完这 5 个之后）

1. **读真正的开源 Flash Attention 实现**：
   - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的 `flash_fwd_kernel.h`
   - 对比你写的简化版，看工业级实现用了哪些技巧（warp-level tiling、Tensor Core、寄存器分块）
2. **读 PagedAttention**：
   - [vLLM](https://github.com/vllm-project/vllm) 的 `csrc/attention/attention_kernels.cu`
   - 理解 KV cache 分页管理和 dynamic batching
3. **学 Triton**：比 CUDA 更易读，能快速写出高性能 kernel，面试很加分
4. **用 Nsight Compute 做 profiling**：学会看 occupancy、memory throughput，这是"真懂 kernel"的标志
5. **把 CosyVoice 的 SDAA 适配层和你的 kernel 学习串起来讲**：
   - "我在 CosyVoice 接入了 FlashAttention，为了深入理解，我自己用 CUDA 实现了简化版"
   - 这条线在面试里非常加分，展示你既能工程落地又能深入理解

---

## 六、文件清单

```
kernel_learning/
├── 01_gelu.cu              # elementwise，入门
├── 02_layernorm.cu         # reduction + online mean/var
├── 03_softmax.cu           # online softmax（Flash Attention 数学基础）
├── 04_gemm.cu              # naive + tiled（面试重点）
├── 05_flash_attention.cu   # 终极目标: tiled GEMM + online softmax 结合
├── reference.py            # CPU 参考实现（本机可跑，已验证全过）
├── Makefile                # 一键编译
└── LEARNING_GUIDE.md       # 本文件
```
