import random
import numpy as np
from hnf_bigint import hnf_bigint
import lattice

for i in range(1000):
    r = random.randint(4, 10)
    n = random.randint(4, 10)

    a = np.random.uniform(-10, 10, size = (r, n)).astype(np.int64)

    res = hnf_bigint(a)

    res2 = lattice.hnf(a)

    assert np.all(res - res2 == 0)

print("ok")

# for a in range(-5, 5):
#     for b in range(-5, 5):
#         if b == 0:
#             continue
#         print(a,b)
#         assert a//b == lattice.int_div(a,b)
