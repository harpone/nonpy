__author__ = 'Heikki Arponen'

import numpy as np
cimport cython
cimport numpy as np

from libc.stdint cimport uint32_t, int32_t
from libc.math cimport sqrt
from libc.math cimport fabs
from libc.math cimport pow

ctypedef float     real_t
ctypedef uint32_t  uint_t
ctypedef int32_t   int_t

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def binner(data, bins):

    """ Put the dX, (dX - drift)(dX - drift) in bins.

    """

    T, dim = data.shape

    dbins_np = (bins[:, 1] - bins[:, 0])  # bin widths; shape = (1, dim)
    # extend bin from start to be consistent with scipy.histogramdd:
    firstcol = bins[:, 0].reshape((dim,1))
    bins = np.hstack((firstcol - dbins_np.reshape((dim, 1)), bins))
    # number of bins:
    nbins = bins.shape[1]
    dbins_np = dbins_np.flatten()
    # to be used in binning the data:
    shiftbins_np = bins[:, 0].flatten()  # shape (dim,)

    cdef:
        np.ndarray[float, ndim=2] data_c = data.astype(np.float32)
        np.ndarray[float, ndim=1] dbins = dbins_np.astype(np.float32) # shape (1, dim)
        np.ndarray[float, ndim=1] shiftbins = shiftbins_np.astype(np.float32) # shape (1, dim)
        # Init histogram, drift and diffusion arrays:
        np.npy_intp size_hist = nbins ** dim
        np.npy_intp size_drift = dim * nbins ** dim
        np.npy_intp size_diff = dim * dim * nbins ** dim
        np.ndarray[uint32_t, ndim=1] hist = np.zeros(size_hist, dtype=np.uint32)
        np.ndarray[float, ndim=1] drifts_unnorm = np.zeros(size_drift, dtype=np.float32)
        np.ndarray[float, ndim=1] diffs_unnorm = np.zeros(size_diff, dtype=np.float32)

        #loop vars etc.
        uint_t t, i, j, drift_location, diff_location
        int_t location, binnumber
        uint_t d = dim
        uint_t T_end = T  # number of samples
        uint_t N = nbins  # number of bins
        float X_ti, dX_ti, dbin, shiftbin
        np.ndarray[float, ndim=1] diffvector = np.zeros(d, dtype=np.float32)

    # Compute histogram and drift array:
    # NOTE: outside bounds values NOT properly accounted!!
    # problem fixed by peeling outer layer of histogram
    for t in range(T_end):
        location = -1
        for i in range(d):
            X_ti = data_c[t,i]
            dbin = dbins[i]
            shiftbin = shiftbins[i]
            binnumber = <int>((X_ti - shiftbin) / dbin)
            if binnumber < 0 or binnumber > N - 1:
                location = -1
                break
            location += <int>(binnumber * pow(N, i))

        if location > -1:
            location += 1
            hist[location] += 1
            for i in range(d):
                drift_location = location + <int>(pow(N, d) * i)
                drifts_unnorm[drift_location] += data_c[t + 1,i] - data_c[t,i]

    # drift array for the diffusion array computation:
    stacked_hist = np.tile(hist, dim)
    cdef:
        np.ndarray[float, ndim=1] drifts = (drifts_unnorm / stacked_hist).astype(np.float32)

    # Compute diffusion array:
    for t in range(T_end):
        location = -1
        for i in range(d):
            X_ti = data_c[t,i]
            dbin = dbins[i]
            shiftbin = shiftbins[i]
            binnumber = <int>((X_ti - shiftbin) / dbin)
            if binnumber < 0 or binnumber > N - 1:
                location = -1
                break
            location += <int>(binnumber * pow(N, i))

        if location > -1:
            location += 1
            for i in range(d):
                dX_ti = data_c[t + 1,i] - data_c[t,i]
                drift_location = location + <int>(pow(N, d) * i)
                diffvector[i] = dX_ti - drifts[drift_location]

            for i in range(d):
                for j in range(d):
                    diff_location = location + <int>(pow(N, d) * (2 * i + j))
                    diffs_unnorm[diff_location] += diffvector[i] * diffvector[j]

    # reshape from 1D arrays:
    new_hist_shape = (nbins,) * dim
    newhist = np.copy(hist).astype(int)
    newhist = newhist.reshape(new_hist_shape)
    # remove outer layer (has out of domain values)
    shave = (slice(1,-1,None),)*dim
    newhist = newhist[shave]

    new_drifts_shape = (dim,) + (nbins,) * dim
    shave = (slice(None, None, None),) + shave
    new_drifts = drifts_unnorm.reshape(new_drifts_shape)
    new_drifts = new_drifts[shave]

    new_diffs_shape = (dim, dim,) + (nbins,) * dim
    shave = (slice(None, None, None),) + shave
    new_diffs = diffs_unnorm.reshape(new_diffs_shape)
    new_diffs = new_diffs[shave]

    # C type arrays have first horizontal axis, then vertical as second
    # so we need to switch x-y coordinates:
    # !!Need to check this in higher dimensions!!
    last_hist = newhist.ndim
    new_hist_axes = np.arange(last_hist)
    new_hist_axes[last_hist - 1], new_hist_axes[last_hist - 2] =\
    new_hist_axes[last_hist - 2], new_hist_axes[last_hist - 1]
    newhist = np.transpose(newhist, new_hist_axes)

    last_drift = new_drifts.ndim
    new_drifts_axes = np.arange(last_drift)
    new_drifts_axes[last_drift - 1], new_drifts_axes[last_drift - 2] =\
    new_drifts_axes[last_drift - 2], new_drifts_axes[last_drift - 1]
    new_drifts = np.transpose(new_drifts, new_drifts_axes)


    last_diff = new_diffs.ndim
    new_diff_axes = np.arange(last_diff)
    new_diff_axes[last_diff - 1], new_diff_axes[last_diff - 2] =\
    new_diff_axes[last_diff - 2], new_diff_axes[last_diff - 1]
    new_diffs = np.transpose(new_diffs, new_diff_axes)


    return new_drifts, new_diffs, newhist
