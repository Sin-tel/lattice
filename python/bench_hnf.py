import numpy as np
import rslattice
from hnf_bigint import hnf_bigint
from time import perf_counter
import random

r = 10
n = 10

a = np.random.uniform(-10, 10, size=(r, n)).astype(np.int64)

t_start = perf_counter()
for i in range(10000):
    res = hnf_bigint(a)
t_stop = perf_counter()
print(f"python {1000*(t_stop - t_start):.2f}ms")


t_start = perf_counter()
for i in range(10000):
    res2 = rslattice.hnf(a)
t_stop = perf_counter()
print(f"rust   {1000*(t_stop - t_start):.2f}ms")
