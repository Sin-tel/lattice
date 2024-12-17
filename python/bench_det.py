import numpy as np
import rslattice
from det import integer_det
from time import perf_counter
import random

n = 20

n_iters = 1000

t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(-10, 10, size=(n, n)).astype(np.int64)
    try:
        res = integer_det(a)
    except OverflowError:
        pass
t_stop = perf_counter()
print(f"python {1000*(t_stop - t_start):.2f}ms")


t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(-10, 10, size=(n, n)).astype(np.int64)
    try:
        res2 = rslattice.integer_det(a)
    except OverflowError:
        pass
t_stop = perf_counter()
print(f"rust   {1000*(t_stop - t_start):.2f}ms")
