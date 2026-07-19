#!/usr/bin/env python3
import subprocess
import sys

import numpy as np


def inputs(m, n, k):
    a = np.array([((i * 17 + 3) % 29 - 14) / 7.0 for i in range(m * k)],
                 dtype=np.float32)
    b = np.array([((i * 11 + 5) % 31 - 15) / 9.0 for i in range(k * n)],
                 dtype=np.float32)
    c = np.array([((i * 7 + 1) % 13 - 6) / 5.0 for i in range(m * n)],
                 dtype=np.float32)
    return a, b, c


def check(executable, m, n, k, alpha, beta, trans_a, trans_b):
    a, b, initial_c = inputs(m, n, k)
    logical_a = a.reshape(k, m).T if trans_a else a.reshape(m, k)
    logical_b = b.reshape(n, k).T if trans_b else b.reshape(k, n)

    output = subprocess.check_output([
        executable, str(m), str(n), str(k), str(alpha), str(beta),
        str(int(trans_a)), str(int(trans_b)),
    ], text=True)
    actual = np.fromstring(output, dtype=np.float32, sep=" ").reshape(m, n)
    expected = (
        np.float32(alpha) * logical_a @ logical_b
        + np.float32(beta) * initial_c.reshape(m, n)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def main():
    executable = sys.argv[1]
    cases = [
        (1, 1, 1, 1.0, 0.0, False, False),
        (2, 5, 3, 0.75, 0.25, False, False),
        (5, 2, 7, 1.0, 0.0, True, False),
        (3, 4, 5, 0.5, 1.5, False, True),
        (4, 3, 6, 1.25, -0.5, True, True),
    ]
    for case in cases:
        check(executable, *case)
    print(f"NumPy GEMM validation passed for {len(cases)} cases")


if __name__ == "__main__":
    main()
