
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
    frac = Fraction(r).limit_denominator(max_den)
    a, b = frac.numerator, frac.denominator

    # 3️⃣ Euler 不和谐度
    return euler_dissonance_from_ratio(a, b)

