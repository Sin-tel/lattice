import numpy as np


# exact integer determinant using Bareiss algorithm
# modified slightly from:
# https://stackoverflow.com/questions/66192894/precise-determinant-of-integer-nxn-matrix
def integer_det(M):
    assert M.ndim == 2
    assert M.shape[0] == M.shape[1]

    M = np.copy(M)  # make a copy to keep original M unmodified

    n, sign, prev = len(M), 1, 1
    for i in range(n - 1):
        if M[i, i] == 0:  # swap with another row having nonzero i's elem
            swapto = next((j for j in range(i + 1, n) if M[j, i] != 0), None)
            if swapto is None:
                return 0  # all M[*][i] are zero => zero determinant
            ## swap rows
            M[[i, swapto]] = M[[swapto, i]]
            sign *= -1
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                # print(M[j, k] * M[i, i] - M[j, i] * M[i, k], prev, (M[j, k] * M[i, i] - M[j, i] * M[i, k]) % prev)
                # if (M[j, k] * M[i, i] - M[j, i] * M[i, k]) % prev != 0:
                #     raise OverflowError("Integer too big for int64")

                d = int(M[j, k]) * int(M[i, i]) - int(M[j, i]) * int(M[i, k])

                assert d % int(prev) == 0
                M[j, k] = d // int(prev)
        prev = M[i, i]
    return sign * M[-1, -1]
