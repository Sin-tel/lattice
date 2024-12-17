import numpy as np
import olll
import lattice
import random

delta = 0.75

for i in range(1000):
    r = random.randint(2, 6)
    n = r + random.randint(0, 4)
    a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
    W = np.eye(n)
    res = olll.reduction(a, delta=delta, W=W)
    res2 = lattice.lll(a, delta, W)
    assert np.all(res - res2 == 0)

print("ok")
