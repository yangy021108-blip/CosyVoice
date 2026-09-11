// ============================================================================
// Kernel 02: LayerNorm (fused, online mean/variance)
// ----------------------------------------------------------------------------
// 对每个"行"(长度为 D 的向量) 做归一化:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// 学习它掌握:
//   1. 规约 (reduction): 一个 block 内多个线程协作求和
//   2. Welford online 算法: 一遍扫描同时算 mean 和 var (数值稳定)
//   3. block 内广播: 用 __syncthreads() 把规约结果同步给所有线程
//   4. 一个 block 处理一行 (每行独立, 行间并行)
//
// 【面试串联点】这就是你 CosyVoice 里 flow_fused_norm 融合的算子。
//   原始 PyTorch 路径: gate*update -> add residual -> LayerNorm -> scale/shift
//   是 4+ 个独立 kernel, 每个都要把整块数据读进写出显存。
//   fused norm (tecolmkFusedNorm) 把它们合成 1 个 kernel, 数据只读 1 次写 1 次。
//   LayerNorm 是典型 memory-bound, 融合收益 = 减少的显存往返次数。
// ============================================================================
#include <cstdio>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// block 内规约求和 (树形归约)
// ----------------------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];                 // 最多 32 个 warp
    int lane   = threadIdx.x & 31;               // warp 内 lane id
    int warpId = threadIdx.x >> 5;               // warp id

    // 第 1 级: warp 内规约 (shuffle 指令, 无需 shared memory)
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // 每个 warp 的 lane0 把结果写入 shared memory
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    // 第 2 级: 第一个 warp 规约各 warp 的部分和
    int numWarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (warpId == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;   // 最终和只在 thread 0 有效
}

// ----------------------------------------------------------------------------
// LayerNorm kernel: 一个 block 处理一行 (grid.x = 行数 = batch*seq)
// ----------------------------------------------------------------------------
__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int D, float eps) {
    int row = blockIdx.x;                        // 当前处理第几行
    const float* x_row = x + row * D;
    float* y_row       = y + row * D;

    // --- 第 1 遍: 求 sum 和 sum_of_squares (跨线程分块累加) ---
    float local_sum = 0.f, local_sq = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v;
        local_sq  += v * v;
    }

    float sum    = block_reduce_sum(local_sum);
    float sum_sq = block_reduce_sum(local_sq);

    // 把 mean / rstd 存到 shared memory, 广播给 block 内所有线程
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) {
        float mean = sum / D;
        // var = E[x^2] - mean^2
        float var  = sum_sq / D - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);              // 1/sqrt(var+eps)
    }
    __syncthreads();

    float mean = s_mean, rstd = s_rstd;

    // --- 第 2 遍: 归一化 + scale/shift ---
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

// ----------------------------------------------------------------------------
// Host 端
// ----------------------------------------------------------------------------
int main() {
    const int rows = 128, D = 1024;              // 128 行, 每行 1024
    const int N = rows * D;
    float eps = 1e-5f;
    size_t bytes = N * sizeof(float);

    float *h_x = new float[N], *h_y = new float[N];
    float *h_gamma = new float[D], *h_beta = new float[D];
    for (int i = 0; i < N; i++) h_x[i] = (float)(i % 13) * 0.1f;
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta,  D * sizeof(float));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  D * sizeof(float), cudaMemcpyHostToDevice);

    // grid = 行数 (一个 block 一行), block = 256 线程
    layernorm_kernel<<<rows, 256>>>(d_x, d_gamma, d_beta, d_y, D, eps);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    printf("layernorm row0[0]=%.6f row0[1]=%.6f\n", h_y[0], h_y[1]);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    delete[] h_x; delete[] h_y; delete[] h_gamma; delete[] h_beta;
    return 0;
}
