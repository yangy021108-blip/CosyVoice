"""
CPU 参考实现 —— 用于验证 4 个 kernel 的算法逻辑 (本机无 GPU 也能跑)
====================================================================
运行: python3 reference.py

这份代码和 .cu 文件一一对应。建议你:
  1. 先跑通这份, 看懂每一步的数学
  2. 再对照 .cu 看 GPU 如何把同样的逻辑并行化
  3. 面试时你能同时讲"算法"(这里) 和"并行化"(.cu), 才算真懂

重点看 online_softmax 和 online_layernorm —— 它们是 Flash Attention 的数学基础。
"""
import math
import random


# ============================================================
# 1. GELU (tanh 近似) —— 对应 01_gelu.cu
# ============================================================
def gelu_tanh(x: float) -> float:
    alpha = math.sqrt(2.0 / math.pi)      # 0.7978845608
    beta = 0.044715
    inner = alpha * (x + beta * x ** 3)
    return 0.5 * x * (1.0 + math.tanh(inner))


# ============================================================
# 2. LayerNorm —— 对应 02_layernorm.cu
# ============================================================
def layernorm(row, gamma, beta, eps=1e-5):
    """row: list[float], 返回归一化后的 list"""
    D = len(row)
    mean = sum(row) / D
    var = sum((v - mean) ** 2 for v in row) / D
    rstd = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * rstd * g + b for v, g, b in zip(row, gamma, beta)]


# ============================================================
# 3. Softmax —— 对应 03_softmax.cu
# ============================================================
def softmax_naive(row):
    """朴素版: 先减 max 防溢出, 再 exp 求和"""
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def online_softmax(row):
    """
    online 版: 一遍扫描维护 (max, sum_exp), 遇到更大 max 就 rescale。
    这正是 CUDA kernel 里 merge_ml 的逻辑, 也是 Flash Attention 的核心。

    递推: 当前状态 (m, l)。新来一个值 v:
      若 v > m:  l = l * exp(m - v) + 1 ; m = v
      否则:      l = l + exp(v - m)
    最后 softmax_i = exp(x_i - m) / l
    """
    m = float('-inf')
    l = 0.0
    for v in row:
        if v > m:
            l = l * math.exp(m - v) + 1.0   # rescale 旧的 sum
            m = v
        else:
            l = l + math.exp(v - m)
    return [math.exp(v - m) / l for v in row]


# ============================================================
# 4. GEMM —— 对应 04_gemm.cu
# ============================================================
def gemm(A, B):
    """朴素三重循环矩阵乘 (理解 tiled 前先懂这个)"""
    N = len(A)
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            acc = 0.0
            for k in range(N):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
    return C


# ============================================================
# 5. Flash Attention —— 对应 05_flash_attention.cu
# ============================================================
def attention_naive(Q, K, V):
    """
    朴素 attention: 先算 N×N 的 S 矩阵, 再 softmax, 再乘 V.
    Q, K, V: list[list[float]], shape [N, D]
    返回: list[list[float]], shape [N, D]
    """
    import math
    N = len(Q)
    D = len(Q[0])
    scale = 1.0 / math.sqrt(D)

    # Step 1: S = Q @ K^T * scale
    S = [[sum(Q[i][d] * K[j][d] for d in range(D)) * scale
          for j in range(N)] for i in range(N)]

    # Step 2: softmax(S, dim=-1)
    P = []
    for i in range(N):
        row_max = max(S[i])
        exps = [math.exp(s - row_max) for s in S[i]]
        row_sum = sum(exps)
        P.append([e / row_sum for e in exps])

    # Step 3: O = P @ V
    O = [[sum(P[i][j] * V[j][d] for j in range(N))
          for d in range(D)] for i in range(N)]
    return O


def attention_flash(Q, K, V, Bc=16):
    """
    Flash Attention 的 CPU 等价实现: 不存 N×N 矩阵, 用 online softmax 分块累加.
    算法和 05_flash_attention.cu 完全对应.

    对每一行 i:
        维护 m_i = -inf, l_i = 0, o_i[D] = 0
        for 每个 K/V 块 (每块 Bc 行):
            s = Q[i] @ K_block^T * scale      (长度 Bc)
            block_max = max(s)
            m_new = max(m_i, block_max)
            rescale = exp(m_i - m_new)
            l_i *= rescale
            o_i *= rescale
            for c in 0..Bc:
                p = exp(s[c] - m_new)
                l_i += p
                o_i += p * V_block[c]
            m_i = m_new
        最后 o_i /= l_i
    """
    import math
    N = len(Q)
    D = len(Q[0])
    scale = 1.0 / math.sqrt(D)
    O = []

    for i in range(N):
        q = Q[i]
        m = float('-inf')
        l = 0.0
        o = [0.0] * D

        # 遍历 K/V 块
        num_kv_blocks = (N + Bc - 1) // Bc
        for kv_block in range(num_kv_blocks):
            k_start = kv_block * Bc
            k_end = min(k_start + Bc, N)

            # Step 1: 计算当前行与 K 块的点积
            s = []
            block_max = float('-inf')
            for c in range(k_start, k_end):
                acc = sum(q[d] * K[c][d] for d in range(D)) * scale
                s.append(acc)
                if acc > block_max:
                    block_max = acc

            # Step 2: online softmax 更新 (核心!)
            m_new = max(m, block_max)
            rescale = math.exp(m - m_new)

            # 一次性 rescale 旧输出和旧 sum
            l *= rescale
            for d in range(D):
                o[d] *= rescale

            # Step 3: 累加新块贡献
            for idx, c in enumerate(range(k_start, k_end)):
                p = math.exp(s[idx] - m_new)
                l += p
                for d in range(D):
                    o[d] += p * V[c][d]

            m = m_new

        # 最终归一化
        inv_l = 1.0 / l
        O.append([v * inv_l for v in o])

    return O


# ============================================================
# 自检
# ============================================================
def _check(name, got, expect, tol=1e-4):
    ok = all(abs(a - b) < tol for a, b in zip(got, expect))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


if __name__ == "__main__":
    random.seed(0)
    all_ok = True

    # --- GELU: 对照几个已知值 ---
    print("GELU:")
    for x in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        print(f"    gelu({x:+.1f}) = {gelu_tanh(x):+.6f}")

    # --- LayerNorm: 均值应为0, 方差应为1 ---
    print("LayerNorm:")
    row = [random.uniform(-3, 3) for _ in range(64)]
    gamma = [1.0] * 64
    beta = [0.0] * 64
    out = layernorm(row, gamma, beta)
    mean = sum(out) / len(out)
    var = sum((v - mean) ** 2 for v in out) / len(out)
    print(f"    归一化后 mean={mean:.6f} (应≈0), var={var:.6f} (应≈1)")

    # --- Softmax: naive 和 online 结果应一致, 且和为1 ---
    print("Softmax (naive vs online):")
    row = [random.uniform(-5, 5) for _ in range(32)]
    naive = softmax_naive(row)
    online = online_softmax(row)
    all_ok &= _check("naive == online", online, naive)
    print(f"    sum(naive)={sum(naive):.6f}  sum(online)={sum(online):.6f} (都应≈1)")

    # --- GEMM: 小矩阵手算校验 ---
    print("GEMM:")
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    C = gemm(A, B)
    expect = [[19.0, 22.0], [43.0, 50.0]]   # 手算结果
    all_ok &= _check("2x2 gemm", [C[0][0], C[0][1], C[1][0], C[1][1]],
                     [expect[0][0], expect[0][1], expect[1][0], expect[1][1]])

    # --- Flash Attention: naive vs flash 应该等价 ---
    print("Flash Attention (naive vs flash):")
    N_attn, D_attn = 32, 8
    Q_small = [[random.uniform(-1, 1) for _ in range(D_attn)] for _ in range(N_attn)]
    K_small = [[random.uniform(-1, 1) for _ in range(D_attn)] for _ in range(N_attn)]
    V_small = [[random.uniform(-1, 1) for _ in range(D_attn)] for _ in range(N_attn)]
    O_naive = attention_naive(Q_small, K_small, V_small)
    O_flash = attention_flash(Q_small, K_small, V_small, Bc=8)
    flat_naive = [v for row in O_naive for v in row]
    flat_flash = [v for row in O_flash for v in row]
    all_ok &= _check("naive == flash (attn)", flat_flash, flat_naive, tol=1e-5)
    print(f"    O_naive[0][0:4] = {[f'{v:.4f}' for v in O_naive[0][:4]]}")
    print(f"    O_flash[0][0:4] = {[f'{v:.4f}' for v in O_flash[0][:4]]}")

    print("\n" + ("全部通过 ✅" if all_ok else "有失败项 ❌"))
    print("\n提示: 看懂这份代码的『算法』, 再对照 .cu 看『并行化』,")
    print("      你就能在面试里同时讲清 math 和 GPU 实现两条线。")
