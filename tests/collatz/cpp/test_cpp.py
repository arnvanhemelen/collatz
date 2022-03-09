#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for C++ module collatz.cpp.
"""

import sys
sys.path.insert(0,'.')

import numpy as np

import collatz.cpp as cpp
import collatz


def run_cloopv(x0, x1, niter):
    """verify cloopv for x in [x0,x1] using niter collatz iterations before checking convergence"""
    
    x = np.arange(x0, x1+1, dtype=int)
    n = np.zeros_like(x)
    nx = x.shape[0]

    print(f"x={x}")
    print(f"n={n}")
    
    cpp.cloopv(x,n,niter)
    
    print(f"after {niter} iterations:")
    print(f"n={n}")
    print("checking the results:")
    for i in range(nx):
        expected = collatz.cloop0(x[i])
        assert(expected == n[i])


def test_cloopv_1_4():
    x0 = 1
    x1 = 4
    # all x values reach 1 in less than 10 iterations
    niter = 10
    run_cloopv( x0, x1, niter )
    
def test_cloopv_1_30():
    x0 = 1
    x1 = 30
    # all x values reach 1 in less than 10 iterations
    niter = 10
    run_cloopv( x0, x1, niter )
    
#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_cloopv_1_30

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
