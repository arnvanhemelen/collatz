# -*- coding: utf-8 -*-

"""Tests for collatz package."""

from re import T
import sys
sys.path.insert(0,'.')

import collatz
import pytest

def test_cloop0():
    """Test for collatz.hello('me')."""
    verbose = True
    assert(collatz.cloop0(3, verbose) == 7 )
    assert(collatz.cloop0(5, verbose) == 5 )
    assert(collatz.cloop0(6, verbose) == 8 )
    assert(collatz.cloop0(7, verbose) == 16 )
    assert(collatz.cloop0(8, verbose) == 3 )
    
def test_cloop():
    for x in [3,5,6,7,8,9,10]:
        assert(collatz.cloop.__wrapped__(x) == collatz.cloop0(x, verbose=True))
    collatz.STOP = 10
    with pytest.raises(RuntimeError):
        x = 9
        assert(collatz.cloop0(x, verbose=True) == collatz.cloop.__wrapped__(x))
    

# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_cloop
    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')

# eof