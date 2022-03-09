# -*- coding: utf-8 -*-

"""
Package collatz
=======================================

Top-level package for our investigation of the collatz conjecture.

https://en.wikipedia.org/wiki/Collatz_conjecture

For any natural number x (i.e. positive integer) 
    if x is even, divide by 2 (x -> x/2)
    if x is odd, multiply by 3 and subtract 1 (x -> 3x+1)
Iterate this infinitely. The Collatz conjecture says that, eventually, 
every number ends up in a 4-2-1 loop. Because 1 is mapped on 3*1+1 = 4, 
4 is mapped on 4/2 = 2 and 2 on 2/2 = 1, once we reach the number 4, we 
are stuck in 4-2-1-4-2-1-4-2-1-...
the number 4 can arise from 
    4 = x/2, or x = 8
and 
    4=3x+1, or x = 1
1 is already in the 4-2-1 loop, so we know that if we reach 8, we are sure
to end up in the loop. Can we reverse the loop further?
    8 = x/2, or x = 16
    8 = 3x+1 or x = 7/3, which is not a natural number.
Hence, once we reach 16, we will end up in 4-2-1.
    16 = x/2, x = 32
    16 = 3x+1, x = 5 
Reversing the iteration we get a first bifurcation after 16 <- (5,32).
Hence, we can stop the iteration at 8.
The reverse of the Collatz conjecture is that by reversing the iteration
we will eventually encounter all possible numbers. Obviously, by construction,
the result of the reverse Collatz loop is always a natural number. But the 
reverse collatz conjecture tells us that we can find any arbitrary natural
number, by iterating backwards.
"""

__version__ = "0.0.0"

try:
    import collatz.cpp
except ModuleNotFoundError as e:
    # Try to build this binary extension:
    from pathlib import Path
    import click
    from et_micc2.project import auto_build_binary_extension
    msg = auto_build_binary_extension(Path(__file__).parent, 'cpp')
    if not msg:
        import collatz.cpp
    else:
        click.secho(msg, fg='bright_red')

#---------------------------------------------------------------------------------------------------
# A simple version of the collatz loop, just for testing that we get things right, can print
# a trace of the iteration
#---------------------------------------------------------------------------------------------------
def cloop0(x, verbose = False):
    """Perform the collatz loop on x and return the number of iterations
    to reach 1.

    There is no mathematical proof that it does, so, it might 
    run forever... However, a counterexample has not been found so far, 
    and if there exists one, it is larger than 2**68...

    :param int x: a positive integer
    :param bool verbose: print a trace of the iterations
    :return: the number of iterations to reach 1. 
    """
    if verbose:
        print(f"collatz.cloop0({x})")
    x0 = x
    
    n = 0
    while x != 1:
        n += 1
        xprev = x
        if x % 2 == 0:
            x = x//2
        else:
            x = 3*x + 1
        if verbose:
            print(f"{n} : {xprev} -> {x}")

    if verbose:
        print()
    else:
        print(f"{x0}->1 in {n} iterations")
    
    return n

#---------------------------------------------------------------------------------------------------
# more advance version of collatz loop:
#   - has a stopping criterion which raise RuntimeError if met
#   - cannot print trace of iteration.
#   - optimized using numba.jit
#---------------------------------------------------------------------------------------------------
STOP = 2**32

from numba import jit

@jit
def cloop(x):
    """Perform the collatz loop on x.
    
    The loop stops at x=8, and therefore this method does NOT work for `x in [4,2,1]`,
    but we don't test for this. 

    :param int x: positive integer number, but not 4, 2, or 1.
    :return: the number of iterations to reach x=1.
    :raises RuntimeError: when the number of iterations exceeds STOP
    """
    n = 0    
    while x != 8:
        n += 1
        x = x//2 if (x%2 == 0) else 3 * x + 1
        if n == STOP:
            print(f"collatz.cloop(): maximum number of iteration reached: {STOP}.")
            raise RuntimeError()
    return n+3 # 3 extra iterations needed to reach 1 (8->4->2->1)

    