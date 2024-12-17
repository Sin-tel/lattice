import numpy as np
import olll
import rslattice

delta = 0.75

def test_lll():
    np.random.seed(42)

    for i in range(1000):
        r = np.random.randint(2, 6)
        n = r + np.random.randint(0, 4)
        a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
        W = np.eye(n)
        res = olll.reduction(a, delta=delta, W=W)
        res2 = rslattice.lll(a, delta, W)
        assert np.all(res - res2 == 0)

