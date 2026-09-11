// ============================================================================
// Kernel 03: Softmax (online / numerically stable)
// ----------------------------------------------------------------------------
// softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
// 学习它掌握:
//   1. 为什么要减 max: 防 exp 溢出 (数值稳定)
//   2. naive 三遍扫描 vs online 一遍扫描的差别
//   3. online softmax 的核心递推 (这是 Flash Attention 的数学基础!)
//
// 【面试高频考点】online softmax:
//   朴素做法需要 3 遍读数据: (1)求max (2)求sum_exp (3)归一化
//   online 做法 1~2 遍: 维护 running (max, sum), 遇到更大的 max 时
//   用 rescale 因子修正已有的 sum, 无需重扫。
//   Flash Attention 正是用 online softmax 实现"分块计算、不存 N*N 矩阵"。
// ============================================================================
#include <cstdio>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// block 内"带 rescale 的规约": 合并 (max, sum_exp) 对
// ----------------------------------------------------------------------------
// 每个线程持有一个局部 (m, l): m=局部max, l=局部sum_exp
// 合并两个 (m1,l1) 和 (m2,l2):
//   m_new = max(m1, m2)
//   l_new = l1*exp(m1-m_new) + l2*exp(m2-m_new)
__device__ __forceinline__ void merge_ml(float& m, float& l,
                                         float other_m, float other_l) {
    if (other_m > m) {
        l = l * expf(m - other_m) + other_l;     // 修正旧的 sum
        m = other_m;
    } else {
        l = l + other_l * expf(other_m - m);
    }
}

__device__ void block_reduce_ml(float& m, float& l) {
    __shared__ float sm[32], sl[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    // warp 内合并
    for (int offset = 16; offset > 0; offset >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, offset);
        float ol = __shfl_down_sync(0xffffffff, l, offset);
        merge_ml(m, l, om, ol);
    }
    if (lane == 0) { sm[warpId] = m; sl[warpId] = l; }
    __syncthreads();

    int numWarps = (blockDim.x + 31) >> 5;
    m = (threadIdx.x < numWarps) ? sm[lane] : -INFINITY;
    l = (threadIdx.x < numWarps) ? sl[lane] : 0.0f;
    if (warpId == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            float om = __shfl_down_sync(0xffffffff, m, offset);
            float ol = __shfl_down_sync(0xffffffff, l, offset);
            merge_ml(m, l, om, ol);
        }
    }
}

// ----------------------------------------------------------------------------
// Online Softmax kernel: 一个 block 处理一行
// ----------------------------------------------------------------------------
__global__ void softmax_online_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int D) {
    int row = blockIdx.x;
    const float* x_row = x + row * D;
    float* y_row       = y + row * D;

    // 每个线程先算自己负责元素的局部 (max, sum_exp)
    float m = -INFINITY, l = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x_row[i];
        merge_ml(m, l, v, 1.0f);                 // 把单个元素并入 (m,l)
    }

    // 跨线程合并得到全局 (max, sum_exp)
    block_reduce_ml(m, l);

    __shared__ float s_m, s_l;
    if (threadIdx.x == 0) { s_m = m; s_l = l; }
    __syncthreads();
    m = s_m; l = s_l;

    // 第 2 遍: 归一化写出
    float inv_l = 1.0f / l;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - m) * inv_l;
    }
}

// ----------------------------------------------------------------------------
// Host 端
// ----------------------------------------------------------------------------
int main() {
    const int rows = 64, D = 512;
    const int N = rows * D;
    size_t bytes = N * sizeof(float);

    float *h_x = new float[N], *h_y = new float[N];
    for (int i = 0; i < N; i++) h_x[i] = (float)(i % 11) * 0.5f - 2.0f;

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    softmax_online_kernel<<<rows, 256>>>(d_x, d_y, D);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    // 校验: 每行之和应约为 1.0
    float rowsum = 0.f;
    for (int i = 0; i < D; i++) rowsum += h_y[i];
    printf("softmax row0 sum = %.6f (应约为 1.0)\n", rowsum);
    printf("softmax row0[0]=%.6f row0[1]=%.6f\n", h_y[0], h_y[1]);

    cudaFree(d_x); cudaFree(d_y);
    delete[] h_x; delete[] h_y;
    return 0;
}
