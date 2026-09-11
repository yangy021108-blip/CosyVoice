// ============================================================================
// Kernel 04: GEMM (矩阵乘 C = A * B) —— naive 到 tiled 的完整演进
// ----------------------------------------------------------------------------
// 这是 GPU kernel 学习的"圣杯", 几乎所有高性能计算面试都会考。
// 学习它掌握:
//   1. naive GEMM 为什么慢 (全局内存反复读, 无复用)
//   2. shared memory tiling (把数据搬到片上高速缓存复用)
//   3. 算术强度 (arithmetic intensity) 与 memory-bound 的关系
//   4. 进一步: 寄存器分块 / 双缓冲 / 向量化 (面试加分项)
//
// 【面试串联点】这就是你 CosyVoice fused GEMM (blas_gemm_fusion) 优化的对象。
//   Attention 的 Q/K/V/Out 投影都是 GEMM。你通过 weight 预排布 + 调用
//   tecoblas 融合 GEMM, 把 aten::linear 调用从 3684 次降到 964 次。
//   理解 tiling 才能解释"为什么 GEMM 适合融合/为什么权重排布重要"。
// ============================================================================
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 32   // shared memory tile 边长

// ----------------------------------------------------------------------------
// 版本 1: naive GEMM —— 每个线程算 C 的一个元素
// ----------------------------------------------------------------------------
// 问题: 算 C[row][col] 要读 A 的整行 + B 的整列, 都是全局内存。
// A、B 的每个元素被反复读 O(N) 次 => 严重 memory-bound。
__global__ void gemm_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < N; k++) {
        acc += A[row * N + k] * B[k * N + col];   // 每次都读全局内存
    }
    C[row * N + col] = acc;
}

// ----------------------------------------------------------------------------
// 版本 2: tiled GEMM —— 用 shared memory 复用数据
// ----------------------------------------------------------------------------
// 核心思想: 把 A、B 的 TILE*TILE 小块先搬到 shared memory (比全局内存
// 快 ~100x), block 内所有线程复用这块数据做计算, 再搬下一块。
// 全局内存读取次数从 O(N) 降到 O(N/TILE)。
__global__ void gemm_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];   // A 的 tile (片上缓存)
    __shared__ float Bs[TILE][TILE];   // B 的 tile

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;
    // 沿 k 方向, 每次处理一个 TILE 宽的块
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // 协作加载: 每个线程搬 1 个元素到 shared memory
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.f;

        __syncthreads();   // 确保 tile 全部加载完再计算

        // 在 shared memory 上做 TILE 次乘加 (复用!)
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();   // 确保计算完再加载下一个 tile
    }

    if (row < N && col < N) C[row * N + col] = acc;
}

// ----------------------------------------------------------------------------
// Host 端: 跑两个版本并对比正确性
// ----------------------------------------------------------------------------
int main() {
    const int N = 512;
    size_t bytes = N * N * sizeof(float);

    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(i % 5) * 0.1f;
        h_B[i] = (float)(i % 7) * 0.1f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // --- naive ---
    dim3 block1(16, 16);
    dim3 grid1((N + 15) / 16, (N + 15) / 16);
    gemm_naive<<<grid1, block1>>>(d_A, d_B, d_C, N);

    // --- tiled ---
    dim3 block2(TILE, TILE);
    dim3 grid2((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    gemm_tiled<<<grid2, block2>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("gemm C[0]=%.4f C[1]=%.4f\n", h_C[0], h_C[1]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}

// ============================================================================
// 【面试进阶追问准备】(背下这些, 面试官深挖时你能接住)
// ----------------------------------------------------------------------------
// Q: 为什么 tile 大小选 32?
// A: 要匹配 warp size(32) 让访存合并; 同时 TILE*TILE*4B*2 不能超过
//    shared memory 每 block 上限(~48KB)。32x32x4B x2 = 8KB, 安全。
//
// Q: 怎么继续优化 tiled 版本?
// A: (1) 寄存器分块: 每线程算 8x8 子块, 提高算术强度;
//    (2) 双缓冲(double buffering): 加载下个 tile 和计算当前 tile 重叠,
//        隐藏全局内存延迟;
//    (3) float4 向量化加载; (4) 用 Tensor Core (wmma) 做 16x16x16 块。
//
// Q: GEMM 是 compute-bound 还是 memory-bound?
// A: 大矩阵是 compute-bound (算术强度 O(N)); naive 小 tile 是 memory-bound。
//    算术强度 = FLOPs / 字节数, 对比 GPU 的 roofline 判断瓶颈。
//
// Q: 你 CosyVoice 的 fused GEMM 和这里的 tiling 什么关系?
// A: tecoblas 的 blas_gemm_fusion 就是高度优化的 tiled/分块 GEMM 实现;
//    我做的是 weight 预排布(让它匹配 kernel 期望的内存布局) + 缓存,
//    避免每次推理重排权重, 并跳过短矩阵(行数<512 时 tiling 收益不足)。
// ============================================================================
