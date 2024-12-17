from time import perf_counter
import time
import numpy as np
import olll
import rslattice


r = 19
n = 20

n_iters = 10

delta = 0.99

W = np.eye(n)

np.random.seed(42)


t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(0, 1000, size=(r, n)).astype(
        np.int64
    )  # this needs to be here for some reason!
    res = olll.reduction(a, delta=delta, W=W)
t_stop = perf_counter()
print(f"python {1000*(t_stop - t_start):.2f}ms")


np.random.seed(42)


t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(0, 1000, size=(r, n)).astype(np.int64)
    res2 = rslattice.lll(a, delta, W)
t_stop = perf_counter()
print(f"rust   {1000*(t_stop - t_start):.2f}ms")
