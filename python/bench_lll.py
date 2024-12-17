from time import perf_counter
import cProfile
import time
import numpy as np
import olll
import lattice


r = 29
n = 30

n_iters = 1


# for i in range(1000):
#     a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
#     W = np.eye(6)
#     res = olll.reduction(a, delta=0.75, W=W)
#     res2 = lattice.lll(a, 0.75, W)
#     assert np.all(res - res2 == 0)

delta = 0.99

W = np.eye(n)

# t_start = perf_counter()
# a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
# res = olll.reduction(a, delta=delta, W=W)
# t_stop = perf_counter()
# print(f"python warmup {1000*(t_stop - t_start):.2f}ms")


t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64) # this needs to be here for some reason!
    res = olll.reduction(a, delta=delta, W=W)
t_stop = perf_counter()
print(f"python {1000*(t_stop - t_start):.2f}ms")


t_start = perf_counter()
for i in range(n_iters):
    a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
    res2 = lattice.lll(a, delta, W)
t_stop = perf_counter()
print(f"rust   {1000*(t_stop - t_start):.2f}ms")

# def main():
#     delta = 0.75

#     a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)
#     W = np.eye(n)

#     t_start = perf_counter()
#     res = olll.reduction(a, delta=delta, W=W)
#     t_stop = perf_counter()
#     print(f"python warmup {1000*(t_stop - t_start):.2f}ms")

#     t_start = perf_counter()
#     for i in range(100):
#         a = np.random.uniform(0, 1000, size = (r, n)).astype(np.int64)

#         res = olll.reduction(a, delta=delta, W=W)
#     t_stop = perf_counter()
#     print(f"python {1000*(t_stop - t_start):.2f}ms")

# cProfile.run('main()')
