__author__ = 'Heikki Arponen'

import numpy as np
cimport cython
cimport numpy as np

from libc.stdint cimport uint32_t, int32_t

ctypedef np.float64_t   real_t
ctypedef uint32_t       uint_t
ctypedef int32_t        int_t


@cython.boundscheck(False)
@cython.wraparound(False)
def corrfun(np.ndarray[real_t, ndim=1] leftarr, np.ndarray[real_t, ndim=1] rightarr, uint_t ran):

    """Helper correlation function. Returns the array defined as
    $$
    C(t) = \sum\limits_{t' = min(0, -t)}^{T - max(0, t)} L(t' + t) R(t'),
    $$
    for $ -ran <= t <= ran$ and where $L=$ leftarr, $R=$ rightarr and $T=$ len(L) = len(R).

    :param leftarr: numpy.float64
    :param rightarr: numpy.float64
    :param ran: int
    :return:
    """
    assert(len(leftarr) == len(rightarr))

    cdef:
        uint_t size = len(leftarr)
        np.ndarray[real_t, ndim=1] corr_array = np.asarray(np.zeros(2*ran + 1), dtype = np.float64)
        uint_t n, m
        real_t temp

    #right hand side and zero:
    for n in range(ran + 1):
        temp = 0
        for m in range(size - n - 1):
            temp = temp + leftarr[m + n] * rightarr[m]
        corr_array[ran + n] = temp
    #left hand side:
    for n in range(1, ran + 1):
        temp = 0
        for m in range(n, size - 1):
            temp = temp + leftarr[m - n] * rightarr[m]
        corr_array[ran - n] = temp

    return corr_array
