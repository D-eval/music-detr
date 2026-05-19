
import math
from fractions import Fraction
import numpy as np

def prime_factorization(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def lcm(a, b):
    return a * b // math.gcd(a, b)


def euler_dissonance_from_ratio(a, b):
    c = lcm(a, b)
    factors = prime_factorization(c)
    return 1 + sum(exp * (p - 1) for p, exp in factors.items())


def pitch_dist_euler(p1, p2, max_den=12):
    # 1️⃣ 频率比（12-TET）
    # return: 1~24
    dp = abs(p1 - p2)
    r = 2 ** (dp / 12)

    # 2️⃣ 有理逼近
    frac = Fraction(float(r)).limit_denominator(max_den)
    a, b = frac.numerator, frac.denominator

    # 3️⃣ Euler 不和谐度
    return euler_dissonance_from_ratio(a, b)


import torch

def build_euler_cost_matrix(freqs, max_den=12, return_log=True):
    """
    freqs: (F,)
    return:
        cost: (F, F)
    """

    freqs = np.asarray(freqs)
    F = len(freqs)

    cost = np.zeros((F, F), dtype=np.float32)

    pitches = 69 + 12 * np.log2(freqs / 440.0)

    for i in range(F):
        for j in range(i, F):

            d = pitch_dist_euler(
                pitches[i],
                pitches[j],
                max_den=max_den
            )

            cost[i, j] = d
            cost[j, i] = d

    cost = torch.tensor(cost)
    eigvals = torch.linalg.eigvalsh(cost)
    # assert eigvals.min() > 0, eigvals

    log_cost = torch.tensor(cost).log()
    cost_m1 = cost - 1
    
    if return_log:
        eigvals = torch.linalg.eigvalsh(log_cost)
        # assert eigvals.min() > 0, eigvals
        return log_cost
    else:
        eigvals = torch.linalg.eigvalsh(cost_m1)
        # assert eigvals.min() > 0, eigvals
        return cost_m1

# loss(A3 | C4) > loss(C4 | A3)
# 根音预测失误，需要有更严重的惩罚

# 我觉得可以直接做一个人为的双向变换
# 也就是人工 embedding pitch_vec
# 那么问题就在于，embedding 之后，用 mse 去接近吗?
# 如果分的足够开，这应该是可以的
# 不行，bce 之所以能够积累训练中的经验，不遗忘，就是因为 sigmoid 允许 logits 很大，对于深信不疑的事情，sigmoid 梯度传回去的很少
# 但是 bce 就没有这个特点
# 也就是说 embedding 空间也必须是 many hot 的形式

# 波形本身可以作为他的表示吗？
# 每个人和他自己的乘积，只和区间长度有关吧，如果边界处为0，也就是说，和他是谁没有关系
# 怎么样让预测能够

# sigmoid + bce 有 logits 累计，不容易遗忘，但是无法规避 incorrect label
# harmony piror 矩阵作为内积空间，能 explict 建模音符距离，但是 many hot 无法收敛，因为 cosine 需要约束到单位圆上
# 这必然导致 norm 的发散，即使加入正则化，无法处理 0 label 的情况
# 就算你用小 norm 去预测 0 label，最终所有 norm 都会降低到分母的 eps 上，导致 cosine 为 0

# detr