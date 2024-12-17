import numpy as np
from hnf_bigint import hnf_bigint
import rslattice

def test_hnf():
    np.random.seed(42)

    for _ in range(1000):
        r = np.random.randint(4, 10)
        n = np.random.randint(4, 10)

        a = np.random.uniform(-10, 10, size = (r, n)).astype(np.int64)

        res = hnf_bigint(a)

        res2 = rslattice.hnf(a)

        assert np.all(res - res2 == 0)

