import collatz
from et_stopwatch import Stopwatch
import numpy as np


if __name__ == "__main__":
    python_cloop = collatz.cloop.__wrapped__
    accelerated_cloop = collatz.cloop

    x0 = 5
    N = 100000

    for block in range(100): # test successive blocks of N elements
        x0 = 5 + block*N
        x1 = x0 + N

        print("\n################################################################################")
        print(f"x=[{x0}, {x1}[")

        
        # Timing the python function
        # print("python_cloop")
        with Stopwatch("python cloop"):
            for x in range(x0,x1):
                python_cloop(x)
        
        # Timing the numba accelerated version
        # print("accelerated_cloop")
        with Stopwatch(" build cloop"):
            # run cloop once to make sure it has been build (numba.jit) before we start timing,
            # to avoid that the build time is included in the timing
            accelerated_cloop(x0)
        with Stopwatch(" numba cloop"):
            for x in range(x0,x1):
                accelerated_cloop(x)

        # Timing the SIMD cpp version
        # print("simd_cloop")
        x = np.arange(x0, x1, dtype=int)
        n = np.zeros_like(x)
        for niter in [2,4,8,16,32,64,128]:
            print(f"niter={niter}")
            with Stopwatch("  SIMD cloop"):
                collatz.cpp.cloopv(x, n, niter)