// ============================================================================
// Kernel 05: Flash Attention (简化版: 单 batch, 单 head)
// ----------------------------------------------------------------------------
// GPU 学习的"终极目标": 把 tiled GEMM (Kernel 04) 和 online softmax
// (Kernel 03) 结合, 实现内存高效的注意力计算.
//
// 【核心思想 - 1 分钟版】
// 朴素 Attention: S = Q @ K^T 是 N×N 矩阵 → 显存 O(N²)
// Flash Attention: 把 Q 按行分块, 对每块遍历 K/V 块,
//   每处理一个 K/V 块就用 online softmax 更新 running (max, sum, output),
//   不再存 N×N 的 S → 显存降到 O(N).
//
// 【算法伪代码 - 面试能复述这个就过关】
//   每个 block 处理 Br 行 Q:
//     每个线程负责 1 行: m = -inf, l = 0, O[D] = 0
//     加载 q_row[D] 到寄存器 (一次, 永久复用)
//     for 每个 K/V 块 (Bc 行):
//         协作加载 K_block[Bc,D], V_block[Bc,D] 到 shared memory
//         计算 s_row[Bc] = q_row @ K_block^T * scale
//         block_max = max(s_row)
//         m_new = max(m, block_max)
//         rescale = exp(m - m_new)
//         O *= rescale              // 一次性 rescale 旧输出
//         l *= rescale              // 一次性 rescale 旧 sum
//         for c in 0..Bc:
//             p = exp(s_row[c] - m_new)
//             l += p
//             O += p * V_block[c, :]
//         m = m_new
//     最终: O /= l, 写回 global memory
//
// 【面试串联点】
// 这就是 CosyVoice Flow DiT 里 _scaled_dot_product_attention 调用的底层算法.
// 你做的 tecocustomFlashAttentionForward 适配层, 背后就是这个逻辑.
// ============================================================================
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// === 配置参数 ===
// 教学用: 小尺寸, 让你能手工验算
#define Br 32      // Q 块行数 = 一个 block 内处理的 output 行数
#define Bc 32      // K/V 块行数 (每轮处理的 K 行数)
#define D  32      // head dimension (每行长度)
                  // 真实 transformer 常是 64/128, 这里为了寄存器压力小, 先小一点
#define WARP 32    // warp size (固定)

// ============================================================================
// 版本 1: Naive Attention (存 N×N 中间矩阵)
// ============================================================================
// 对照用. 显存 O(N²), 大 sequence 直接 OOM.
__global__ void attention_naive(
    const float* Q, const float* K, const float* V, float* O,
    int N, float scale)
{
    // 一个 block 算一行 O
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float s_S[];    // [N] 存一行 S

    // Step 1: S[row, j] = Q[row] @ K[j]^T * scale
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float acc = 0.f;
        for (int d = 0; d < D; d++) {
            acc += Q[row * D + d] * K[j * D + d];
        }
        s_S[j] = acc * scale;
    }
    __syncthreads();

    // Step 2: softmax (两遍: 找 max, 然后 exp 求和)
    // --- 找 max ---
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, s_S[j]);
    // 跨线程规约
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float row_max = s_max;

    // --- exp + 求和 ---
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        s_S[j] = expf(s_S[j] - row_max);
        local_sum += s_S[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    float row_sum = s_sum;

    // Step 3: O[row, d] = sum_j (S[row,j]/sum) * V[j, d]
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < N; j++) {
            acc += (s_S[j] / row_sum) * V[j * D + d];
        }
        O[row * D + d] = acc;
    }
}


// ============================================================================
// 版本 2: Flash Attention (分块 + online softmax, 不存 N×N 矩阵)
// ============================================================================
// 设计:
//   - 一个 block 处理 Br 行输出 (Br 个线程, 每线程 1 行)
//   - 每线程把 Q 行加载到寄存器, 然后迭代处理 K/V 块
//   - 每轮用 online softmax 更新 running (m, l, O)
//   - 最终 O 除以 l 得到答案
//
// 共享内存布局:
//   K_shared [Bc, D]    —— 当前 K 块
//   V_shared [Bc, D]    —— 当前 V 块
// 总大小: 2 * 32 * 32 * 4B = 8192 B = 8 KB (完全 OK)
//
// 每个线程的寄存器状态:
//   q[D] = 32 floats        —— 自己的 Q 行 (永久持有, 反复用)
//   o[D] = 32 floats        —— 当前输出累加器
//   s[Bc] = 32 floats       —— 当前 K/V 块的 S 行 (临时)
//   m, l = 2 floats         —— running max, running sum of exp
// 总计 ~100 寄存器/线程, 完全可接受
__global__ void attention_flash(
    const float* __restrict__ Q,   // [N, D]
    const float* __restrict__ K,   // [N, D]
    const float* __restrict__ V,   // [N, D]
    float* __restrict__ O,         // [N, D]
    int N, float scale)
{
    // --- 身份识别 ---
    int block_q_start = blockIdx.x * Br;       // 本 block 负责的 Q 起始行
    int my_row_in_block = threadIdx.x;         // 0..Br-1, 本线程负责的行
    int my_global_row = block_q_start + my_row_in_block;
    bool active = (my_global_row < N);

    // --- Shared memory ---
    __shared__ float K_shared[Bc][D];
    __shared__ float V_shared[Bc][D];

    // --- 每线程寄存器状态 (Flash Attention 的核心!) ---
    float q[D];        // 我的 Q 行, 加载一次, 全程复用
    float o[D];        // 输出累加器, 初始化为 0
    float m = -INFINITY;   // running max
    float l = 0.f;         // running sum of exp

    // 加载 Q 行到寄存器 (每行只读 global memory 一次!)
    if (active) {
        for (int d = 0; d < D; d++) {
            q[d] = Q[my_global_row * D + d];
            o[d] = 0.f;
        }
    }

    // ========= 主循环: 遍历所有 K/V 块 =========
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int block_k_start = kv_block * Bc;

        // --- 协作加载 K_block 和 V_block 到 shared memory ---
        // Br 个线程共同加载 Bc*D 个元素 (每线程加载 Bc*D/Br 个)
        for (int idx = my_row_in_block; idx < Bc * D; idx += Br) {
            int r = idx / D, c = idx % D;
            int global_r = block_k_start + r;
            if (global_r < N) {
                K_shared[r][c] = K[global_r * D + c];
                V_shared[r][c] = V[global_r * D + c];
            } else {
                K_shared[r][c] = 0.f;
                V_shared[r][c] = 0.f;
            }
        }
        __syncthreads();   // 确保 shared memory 加载完成

        if (active) {
            // --- Step 1: 计算本行与当前 K 块的点积 S_row[0..Bc-1] ---
            //   S_row[c] = q · K_shared[c] · scale
            float s[Bc];
            float block_max = -INFINITY;
            for (int c = 0; c < Bc; c++) {
                float acc = 0.f;
                for (int d = 0; d < D; d++) {
                    acc += q[d] * K_shared[c][d];
                }
                s[c] = acc * scale;
                if (s[c] > block_max) block_max = s[c];
            }

            // --- Step 2: Online softmax 更新 (核心!) ---
            // 新 max 可能是旧的 m, 也可能是当前块的 block_max
            float m_new = fmaxf(m, block_max);

            // rescale = exp(m_old - m_new): 旧输出要乘这个系数
            // 如果 m_new == m (max 没变), rescale = 1.0, 什么都没发生
            // 如果 m_new > m (max 变大了), rescale < 1.0, 旧输出被压缩
            float rescale = expf(m - m_new);

            // 一次性 rescale 旧的 O 和旧的 l (关键! 只做 1 次, 不是 Bc 次)
            l *= rescale;
            for (int d = 0; d < D; d++) o[d] *= rescale;

            // --- Step 3: 累加新块的贡献 ---
            // 对每个 K/V 位置 c:
            //   p = exp(S[c] - m_new)   -- 未归一化的 attention weight
            //   l += p                  -- running sum
            //   O += p * V[c]           -- 加到输出
            for (int c = 0; c < Bc; c++) {
                int global_k = block_k_start + c;
                if (global_k >= N) continue;   // 边界外: K/V 都是 0, 跳过

                float p = expf(s[c] - m_new);
                l += p;
                for (int d = 0; d < D; d++) {
                    o[d] += p * V_shared[c][d];
                }
            }

            // 更新 running max
            m = m_new;
        }
        __syncthreads();   // 确保所有线程用完 shared memory 再覆盖
    }

    // ========= 最终归一化: O /= l =========
    if (active) {
        float inv_l = 1.f / l;
        for (int d = 0; d < D; d++) {
            O[my_global_row * D + d] = o[d] * inv_l;
        }
    }
}


// ============================================================================
// Host 端: 跑两个版本并对比正确性
// ============================================================================
int main() {
    const int N = 128;           // sequence length
    const float scale = 1.0f / sqrtf((float)D);
    const size_t qkv_bytes = (size_t)N * D * sizeof(float);

    // --- Host 内存 ---
    float *h_Q = new float[N*D], *h_K = new float[N*D], *h_V = new float[N*D];
    float *h_O_naive = new float[N*D], *h_O_flash = new float[N*D];

    // 初始化: 小随机数, 避免 softmax 数值问题
    for (int i = 0; i < N*D; i++) {
        h_Q[i] = (float)((i * 17 + 3) % 11 - 5) * 0.1f;
        h_K[i] = (float)((i * 23 + 7) % 13 - 6) * 0.1f;
        h_V[i] = (float)((i * 31 + 11) % 9 - 4) * 0.1f;
    }

    // --- Device 内存 ---
    float *d_Q, *d_K, *d_V, *d_O_n, *d_O_f;
    cudaMalloc(&d_Q, qkv_bytes);
    cudaMalloc(&d_K, qkv_bytes);
    cudaMalloc(&d_V, qkv_bytes);
    cudaMalloc(&d_O_n, qkv_bytes);
    cudaMalloc(&d_O_f, qkv_bytes);
    cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice);

    // --- 跑 naive (每个 block 算 1 行, 需要 N 行 shared memory) ---
    attention_naive<<<N, WARP, N * sizeof(float)>>>(d_Q, d_K, d_V, d_O_n, N, scale);

    // --- 跑 flash (每个 block 算 Br 行, Br 个线程) ---
    int num_blocks = (N + Br - 1) / Br;
    attention_flash<<<num_blocks, Br>>>(d_Q, d_K, d_V, d_O_f, N, scale);

    // --- 取回结果并对比 ---
    cudaMemcpy(h_O_naive, d_O_n, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_flash, d_O_f, qkv_bytes, cudaMemcpyDeviceToHost);

    float max_diff = 0.f;
    float sum_diff = 0.f;
    for (int i = 0; i < N*D; i++) {
        float diff = fabsf(h_O_naive[i] - h_O_flash[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    printf("N=%d, D=%d, Br=%d, Bc=%d\n", N, D, Br, Bc);
    printf("naive vs flash max  diff = %.6e (应接近 0)\n", max_diff);
    printf("naive vs flash mean diff = %.6e\n", sum_diff / (N*D));
    printf("Sample: naive O[0:4] = [%.4f %.4f %.4f %.4f]\n",
           h_O_naive[0], h_O_naive[1], h_O_naive[2], h_O_naive[3]);
    printf("Sample: flash O[0:4] = [%.4f %.4f %.4f %.4f]\n",
           h_O_flash[0], h_O_flash[1], h_O_flash[2], h_O_flash[3]);

    // --- 释放 ---
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_n); cudaFree(d_O_f);
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_naive; delete[] h_O_flash;
    return 0;
}


// ============================================================================
// 【关键理解检查清单】(自问自答这 5 个问题, 全答对 = 真懂了)
// ----------------------------------------------------------------------------
//
// 1. 为什么 Q 行只读一次 global memory, 但 K/V 要读很多次?
//    因为 Q 行被这个线程永久持有 (寄存器 q[D]), 每个 KV 块都用它点乘.
//    K/V 分块, 每块只读一次 shared memory, 但不同 KV 块要反复读 global memory.
//    → 这就是"tiling 沿哪个轴"的选择. 选 Q 轴是因为一个 block 处理多行 Q,
//      每行 Q 需要和所有 K/V 交互, 所以 Q 常驻.
//
// 2. 为什么 rescale 必须在 for c 循环外面, 不能放里面?
//    因为 rescale = exp(m - m_new) 是针对"旧 O"的, 跟当前块的 c 无关.
//    如果放在内层: 第 1 次 c 迭代: O = O * rescale + p1 * V1 ✓
//                  第 2 次 c 迭代: O = (O * rescale + p1 * V1) * rescale + p2 * V2
//                               = O * rescale² + p1*V1*rescale + p2*V2 ✗ 错!
//    正确: O = O * rescale 一次, 然后累加所有 p_c * V_c.
//    → 这就是为什么代码里 "l *= rescale; for d: o[d]*=rescale" 在 for c 外面.
//
// 3. Flash Attention 比 naive 快吗?
//    速度差不多 (FLOPs 相同), 但显存从 O(N²) 降到 O(N).
//    实际更快是因为: (a) 减少 HBM 读写 (b) 允许更大 batch/sequence.
//    面试回答: "Flash 的优势在显存, 不在 FLOPs".
//
// 4. 你的 CosyVoice 里 Flash Attention 和这个有什么差别?
//    工业版: 多线程协作处理一行 Q (warp-level tiling, 提高并行度);
//            用 Tensor Core / wmma 加速 GEMM 部分;
//            处理多 batch / 多 head / causal mask / dropout / 变长;
//            支持 bf16/fp16 混合精度.
//    教学版: 单线程一行, 单 batch, 单 head, 无 mask, fp32.
//    → 算法核心 (online softmax + rescale) 完全一样!
//
// 5. Causal mask 怎么加?
//    在 "for c in Bc" 循环里加条件:
//       if (block_k_start + c > my_global_row) {
//           // 这个位置不能看, 跳过 (相当于 p = 0)
//           continue;
//       }
//    或者更优雅: 在计算 s[c] 时就写 s[c] = -INFINITY 对于未来位置,
//    这样 exp(s[c] - m_new) = exp(-inf) = 0, 自动被 mask 掉.
// ============================================================================


// ============================================================================
// 【性能数字估算】(面试加分: 能算出来 = 真懂)
// ----------------------------------------------------------------------------
// 用你的 CosyVoice 实际配置: N=512 (典型 Flow DiT 序列长), head_dim=64
//
// Naive:
//   中间矩阵 S: 512 × 512 × 4B = 1 MB (per head per batch)
//   如果 batch=8, heads=8, layers=20: 8 × 8 × 20 × 1 MB = 1.28 GB
//   (只是 S, 还没算 softmax 的中间值)
//
// Flash Attention:
//   S_block: Br × Bc × 4B = 32 × 32 × 4 = 4 KB (per block, 用完即丢)
//   running stats: 每个 block Br × (D + 2) × 4B ≈ 4 KB
//   总计: 8 × 8 × 20 × (4KB + 4KB) × (512/32 blocks) = 130 MB
//
// 差距: 1.28 GB vs 130 MB ≈ 10×. 大 N 时差距更大 (O(N²) vs O(N)).
// ============================================================================
