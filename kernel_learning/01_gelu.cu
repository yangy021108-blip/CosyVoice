// ============================================================================
// Kernel 01: GELU (elementwise activation)
// ----------------------------------------------------------------------------
// 这是最简单的 GPU kernel 类型:每个输出元素只依赖一个输入元素,元素之间
// 没有依赖关系。学习它掌握:
//   1. kernel launch 配置 (grid/block 如何划分)
//   2. 全局内存的"合并访存"(coalesced access)
//   3. 边界处理 (当 N 不是 blockDim 整数倍时)
//
// 【面试串联点】这就是你在 CosyVoice fused FFN 里用到的激活函数。
//   原始路径: Linear -> GELU -> Linear (3 个 kernel, 2 次中间读写显存)
//   fused 路径: ffnv2 把三者融成 1 个 kernel, 省掉中间 tensor 的显存读写。
//   GELU 本身是 memory-bound (计算量小、访存量大), 融合的价值正在于此。
// ============================================================================
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// tanh 近似版 GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// 相比精确版 erf, tanh 版在 GPU 上更快且精度足够。
__device__ __forceinline__ float gelu_tanh(float x) {
    const float kAlpha = 0.7978845608f;   // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float inner = kAlpha * (x + kBeta * x * x * x);
    // tanh 用硬件指令实现, 比 exp 快
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ----------------------------------------------------------------------------
// Kernel 主体: 每个线程处理 1 个元素
// ----------------------------------------------------------------------------
__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int N) {
    // 全局线程 id = blockIdx.x * blockDim.x + threadIdx.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界保护: 启动的线程数 >= N 时, 多余线程直接返回
    if (idx >= N) return;

    // 核心: 1 读 + 1 写 + 少量计算
    out[idx] = gelu_tanh(in[idx]);
}

// ----------------------------------------------------------------------------
// 进阶版: 每个线程处理 4 个元素 (float4 向量化访存)
// ----------------------------------------------------------------------------
// 【为什么更快】GPU 内存事务以 32/128 字节为单位。float4 一次搬 16 字节,
// 减少指令数并提高内存吞吐。这是 elementwise kernel 最常用的优化。
__global__ void gelu_kernel_vec4(const float4* __restrict__ in,
                                 float4* __restrict__ out,
                                 int N4) {           // N4 = N / 4
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    float4 v = in[idx];               // 一次读 4 个 float
    v.x = gelu_tanh(v.x);
    v.y = gelu_tanh(v.y);
    v.z = gelu_tanh(v.z);
    v.w = gelu_tanh(v.w);
    out[idx] = v;                     // 一次写 4 个 float
}

// ----------------------------------------------------------------------------
// Host 端: 分配、初始化、launch、校验
// ----------------------------------------------------------------------------
int main() {
    const int N = 1 << 20;            // ~1M 元素
    size_t bytes = N * sizeof(float);

    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 7) - 3.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // launch 配置: 每 block 256 线程, grid 向上取整
    const int block = 256;
    int grid = (N + block - 1) / block;
    gelu_kernel<<<grid, block>>>(d_in, d_out, N);

    // 向量化版本 (要求 N % 4 == 0)
    // int grid4 = (N/4 + block - 1) / block;
    // gelu_kernel_vec4<<<grid4, block>>>((float4*)d_in, (float4*)d_out, N/4);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 校验前几个元素 (对照 CPU: gelu_tanh)
    for (int i = 0; i < 5; i++) {
        float cpu = gelu_tanh(h_in[i]);  // 这里直接调 device 函数的 host 版本需另写, 仅示意
        printf("in=%+.2f  gpu_out=%+.6f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
