# Collatz iteration

## Approach

1. **`collatz.cloop` Simple Python only version**. probably inefficient, but useful to validate later improvements. Takes 2.151362s to test the first 100 000 integers starting from 5. 

2. **`collatz.cloop` with numba acceleration**. Nice improvement, takes only 0.082121s, i.e. 26.2 times faster. 

3. **`collatz.cpp.cloopv`**, a C++ implementation allowing for vectorization, i.e. allows to apply collatz iteration to 4 integers at once, instead of 1. This is useful because we want to test as many 
   numbers as possible. The function accepts three parameters:
    * a numpy array `x` with the numbers to be tested,
    * a numpy array `n` which on return contains the number of iteration before reaching 1,
    * an integer `niter` that determines how many iteration are applied before checking for convergence and replacing converged entries with the next number to be checked. I expected it to be 
    less efficient to check every iteration. (This will turn out to be wrong!)

   `collatz.cpp.cloopv` improves on the numba accelerated version by a factor of 2 or more. 

4. **`collatz.cpp.cloopv1`**: a version of **`collatz.cpp.cloopv`** with `niter = 1`. This turns out to improve by another factor of almost 2. 

5. **`collatz.cpp.cloopv1b`**: a version of **`collatz.cpp.cloopv1`** where the main loop performs no test to check if x == 1 already. This requires a separate loop for the (final) case where the input array is exhausted and there are no more values to treat. Here, the test cannot be avoided, as otherwise the the collatz loop would be applied even if the entry has already converged. This reduces the runtime to 0.014616 s. 

Here are the timings:

    cloop      : 2.1514s
    numba cloop: 0.0821s x26.2
    cloopv     : 0.0350s x61.5
    cloopv1    : 0.0245s x86.3
    cloopv1b   : 0.0146s x147.2

These are impressive results, but, nevertheless, some important lessons are to be learned.

## Lessons learned...

1. The assumption that the vectorized cloopv would be faster if it can perform many iterations before replacing a converged entry turned out to be wrong. Gut feelings about efficiency are often wrong. This is called **premature optimization**. It also violates the KIS principle: Keep It Simple. Conceptually, `collatz.cpp.cloopv1` is simpler than `collatz.cpp.cloopv` and should have been tried out first. In fact, a bit of research and inspection of the vectorization reports was needed to get the loop vectorized. Autovectorization can only vectorize the inner loop and prefers to vectorize the largest loop. Therefor the compiler switches the loops over `niter` and `N`, so that the `niter` loop is the innerloop.  Next, it finds out that the `niter` loop is not vectorizable because of dependencies, and gives up.  The problem can be solved by putting a `#pragma simd` before the `N` loop to force the compiler to vectorize the `N` loop.

   Instead of following gut feelings experimentation and measurement must be the basis of our improvements. 

2. Numba gives us a a quick and big improvement, but our C++ implementation provides an additional speedup of 5.6 times. This is more than the factor of 4 we can expect from vectorisation. This shows that the non-vectorized Numba version could be better, and that it is not vectorized. In fact a trick was needed to allow the vectorisation: the use of the work array `xv` that can just fill a vector register. 