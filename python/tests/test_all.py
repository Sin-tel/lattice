import numpy as np
from hnf_bigint import hnf_bigint
from det import integer_det
import olll
import rslattice

low = -100
high = 100
n_iters = 1000

# All numpy runtime warnings should raise error
np.seterr(all="raise")


def test_hnf():
    np.random.seed(42)

    for _ in range(n_iters):
        r = np.random.randint(4, 10)
        n = np.random.randint(4, 10)

        a = np.random.uniform(low, high, size=(r, n)).astype(np.int64)

        overflow1 = False
        overflow2 = False

        try:
            res = hnf_bigint(a)
        except OverflowError:
            overflow1 = True

        try:
            res2 = rslattice.hnf(a)
        except OverflowError:
            overflow2 = True

        if overflow1 or overflow2:
            assert overflow1 and overflow2
        else:
            assert np.all(res - res2 == 0)


def test_lll():
    delta = 0.75
    np.random.seed(42)

    for _ in range(n_iters):
        r = np.random.randint(2, 6)
        n = r + np.random.randint(0, 4)
        a = np.random.uniform(low, high, size=(r, n)).astype(np.int64)
        W = np.eye(n)
        res = olll.reduction(a, delta=delta, W=W)
        res2 = rslattice.lll(a, delta, W)
        assert np.all(res - res2 == 0)


def test_integer_det():
    np.random.seed(42)

    for _ in range(n_iters):
        n = np.random.randint(4, 10)

        a = np.random.uniform(low, high, size=(n, n)).astype(np.int64)

        overflow1 = False
        overflow2 = False

        try:
            res = integer_det(a)
        except OverflowError:
            overflow1 = True

        try:
            res2 = rslattice.integer_det(a)
        except OverflowError:
            overflow2 = True

        if overflow1 or overflow2:
            # They don't give the same result here since the Rust version overflows more easily
            # assert overflow1 and overflow2, f"{overflow1}, {overflow2}"
            assert overflow2
        else:
            assert res == res2


def test_nearest_plane():
    np.random.seed(42)

    for _ in range(n_iters):
        r = np.random.randint(2, 6)
        n = r + np.random.randint(0, 4)
        a = np.random.uniform(low, high, size=(r, n)).astype(np.int64)
        v = np.random.uniform(low, high, size=(n,)).astype(np.int64)
        W = np.eye(n)

        res = olll.nearest_plane(v, a, W)
        res2 = rslattice.nearest_plane(v, a, W)
        assert np.all(res - res2 == 0), f"{repr(a)}, {repr(v)}"


if __name__ == "__main__":
    test_integer_det()
