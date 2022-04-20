/*
 *  C++ source file for module collatz.cpp
 */


// See http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html for examples on how to use pybind11.
// The example below is modified after http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html#More-on-working-with-numpy-arrays
#include <cstdint>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#include "my_array_t.hpp"

// This method aims at a SIMD version of the collatz loop
// 
void
cloopv
  ( py::array_t<int64_t> x_ // numpy array with x values, updated by the algorithm until all are 1
  , py::array_t<int64_t> n_ // numpy array with the iteration count. initially all 0
  , int64_t niter           // number of iterations that is run before checking for converged entries
  )
{
    my::array_t<int64_t,1> x__(x_);
    my::array_t<int64_t,1> n__(n_);
    int const nx = x__.shape(0);
    int64_t* x = x__.data();
    int64_t* n = n__.data();

 // define work arrays whose length corresponds to a vector register (4 64-bit words)
    int const N = 4;
    assert( N <= nx && "Expecting at least N input values" ); 
    int64_t xv[N], nv[N];
    int     jv[N]; 
 // initialize them with the first N items in the input array x_
    int j = 0;
    for( ; j < N; ++j ) {
        xv[j] = x[j]; // copy an element from the x array to the work array
        nv[j] = 0;    // reset the iteration counter 
        jv[j] = j;    // remember which element of the x array is being treated here
    }

    bool done = false; 
    while( !done )
    {// Perform niter collatz iterations 
        for( int64_t iter = 0; iter < niter; ++iter )
        {// The auto-vectorization interchanges the loop order (because the outer loop is longer),
         // than finds out that there is a dependency, and gives up. Adding #pragma simd prevents
         // the loop interchange, and vectorizes the short inner loop
            #pragma simd
            for( int i = 0; i < N; ++i ) {
                if( xv[i] > 1 ) {
                    xv[i] = ( xv[i] % 2 == 0 ? xv[i] / 2 : 3 * xv[i] + 1 );
                    nv[i] += 1;
                }
            }
        }
     // check which elements of xv have reached 1, and replace them with the next value from the x array:
        for( int i = 0; i < N; ++i ) {
            if( xv[i] == 1 ) {
                n[jv[i]] = nv[i];   // store the result in the output array
                //std::cout<<"2: i="<<i<<", jv[i]="<<jv[i]<<", n[jv[i]]="<<n[jv[i]]<<std::endl;
                if( j < nx ) {      // not all x values have been processed so far.
                    xv[i] = x[j];   // fetch a new x value
                    nv[i] = 0;      // reset the corresponding iteration count
                    jv[i] = j;      // remember which element we are treating here
                    ++j;            // increment j,to point to the next element to be processed.
                }
            }
        }
     // we are done if we exhausted the x array:
        done = (j == nx);
     // and if all elements in xv have reached 1:
        for( int i = 0; i < N; ++i ) 
            done &= (xv[i] == 1);
    }
}

// A version of cloopv with niter=1
void
cloopv1
  ( py::array_t<int64_t> x_ // numpy array with x values, updated by the algorithm until all are 1
  , py::array_t<int64_t> n_ // numpy array with the iteration count. initially all 0
  )
{
    my::array_t<int64_t,1> x__(x_);
    my::array_t<int64_t,1> n__(n_);
    int const nx = x__.shape(0);
    int64_t* x = x__.data();
    int64_t* n = n__.data();

 // define work arrays whose length corresponds to a vector register (4 64-bit words)
    int const N = 4;
    assert( N <= nx && "Expecting at least N input values" ); 
    int64_t xv[N], nv[N];
    int     jv[N]; 
 // initialize them with the first N items in the input array x_
    int j = 0;
    for( ; j < N; ++j ) {
        xv[j] = x[j]; // copy an element from the x array to the work array
        nv[j] = 0;    // reset the iteration counter 
        jv[j] = j;    // remember which element of the x array is being treated here
    }

    bool done = false;
    while( !done )
    {
        for( int i = 0; i < N; ++i ) 
        {// Perform a single collatz iteration:
            if( xv[i] > 1 ) {
                xv[i] = ( xv[i] % 2 == 0 ? xv[i] / 2 : 3 * xv[i] + 1 );
                nv[i] += 1;
            }
         // check which elements of xv have reached 1, and replace them with the next value from the x array:
            if( xv[i] == 1 )
            {// store the result in the output array:
                n[jv[i]] = nv[i];
                //std::cout<<"2: i="<<i<<", jv[i]="<<jv[i]<<", n[jv[i]]="<<n[jv[i]]<<std::endl;
                if( j < nx ) {      // not all x values have been processed so far.
                    xv[i] = x[j];   // fetch a new x value
                    nv[i] = 0;      // reset the corresponding iteration count
                    jv[i] = j;      // remember which element we are treating here
                    ++j;            // increment j,to point to the next element to be processed.
                } else 
                {// there are no values left to copy to the work array. We put a zero instead, so, the
                // entry is no longer processed nor stored again in the output array.
                    xv[i] = 0;
                }
            }
        }
     // we are done if we exhausted the x array:
        done = (j == nx);
     // and if all elements in xv have reached 1 (in which case they have been replaced with 0)
        for( int i = 0; i < N; ++i ) 
            done &= (xv[i] == 0);
    }
}


// A version of cloopv with niter=1 that does not have to test for xv[i] > 1 in the main loop.
// The consequence is that we must implement a separate loop for when there is less than 4 
// entries in the work array (input array exhausted).
// caveat: cloopv1b always performs at least one collatz iteration, so it fails to produce
// the correct number of iterations for x=1.
void
cloopv1b
  ( py::array_t<int64_t> x_ // numpy array with x values, updated by the algorithm until all are 1
  , py::array_t<int64_t> n_ // numpy array with the iteration count. initially all 0
  )
{
    my::array_t<int64_t,1> x__(x_);
    my::array_t<int64_t,1> n__(n_);
    int const nx = x__.shape(0);
    int64_t* x = x__.data();
    int64_t* n = n__.data();

 // define work arrays whose length corresponds to a vector register (4 64-bit words)
    int const N = 4;
    assert( N <= nx && "Expecting at least N input values" ); 
    int64_t xv[N], nv[N];
    int     jv[N]; 
 // initialize them with the first N items in the input array x_
    int j = 0;
    for( ; j < N; ++j ) {
        xv[j] = x[j]; // copy an element from the x array to the work array
        nv[j] = 0;    // reset the iteration counter 
        jv[j] = j;    // remember which element of the x array is being treated here
    }

    bool done = false;
    while( !done )
    {
        for( int i = 0; i < N; ++i ) 
        {// Perform a single collatz iteration:
            xv[i] = ( xv[i] % 2 == 0 ? xv[i] / 2 : 3 * xv[i] + 1 );
            nv[i] += 1;
         // check which elements of xv have reached 1, and replace them with the next value from the x array:
            if( xv[i] == 1 )
            {// store the result in the output array:
                n[jv[i]] = nv[i];
                //std::cout<<"2: i="<<i<<", jv[i]="<<jv[i]<<", n[jv[i]]="<<n[jv[i]]<<std::endl;
                if( j < nx ) {      // not all x values have been processed so far.
                    xv[i] = x[j];   // fetch a new x value
                    nv[i] = 0;      // reset the corresponding iteration count
                    jv[i] = j;      // remember which element we are treating here
                    ++j;            // increment j,to point to the next element to be processed.
                }
                // else 
                // {// there are no values left to copy to the work array. We put a zero instead, so, the
                // // entry is no longer processed nor stored again in the output array.
                //     xv[i] = 0;
                // }
            }
        }
     // we are done if we exhausted the x array:
        done = (j == nx);
    //  // and if at least one of the entries is converged to 1
    //     bool at_least_one_converged_entry = (xv[0] == 1);
    //     for( int i=1; i<N; ++i ) {
    //         at_least_one_converged_entry |= (xv[i] == 1);
    //     }
    }
 // iterate the last N entries

    int n_left = N;
    for( int i=0; i<N; ++i ) {
        if( xv[i] == 1 ) 
            n_left -= 1;
        // std::cout<<"end0: x="<<xv[i]<<" j="<<jv[i]<<std::endl;
    }
    while (n_left > 0) 
    {
        for( int i=0; i<N; ++i )
        {
            // std::cout<<"end : "<<i<<":"<<xv[i]<<std::endl;
            if( xv[i] > 1 ) // otherwise the entry is already converged
            {// Perform a single collatz iteration:
                xv[i] = ( xv[i] % 2 == 0 ? xv[i] / 2 : 3 * xv[i] + 1 );
                nv[i] += 1;
            // test again for convergence
                if( xv[i] == 1 ) 
                {// store the result in the output array:
                    n[jv[i]] = nv[i];
                    n_left -= 1;
                }
            }
        }
    }
}


PYBIND11_MODULE(cpp, m)
{// optional module doc-string
    m.doc() = "pybind11 cpp plugin"; // optional module docstring
 // list the functions you want to expose:
 // m.def("exposed_name", function_pointer, "doc-string for the exposed function");
    m.def("cloopv", &cloopv);
    m.def("cloopv1", &cloopv1);
    m.def("cloopv1b", &cloopv1b);
}
