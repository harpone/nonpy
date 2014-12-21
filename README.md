nonpy
=====

Nonparametric time series modeling.

Fit multivariate data to a diffusion process.

Pre-alpha stage (i.e. almost completely useless so far).

Currently the only useful function is the (Cython 
optimized) 'crosscorrelate' function, which does cross-
correlation (with lag) between two Pandas time series. 
You could of course use scipy.correlate, but that will
be very slow for large datasets where you only want
to look correlations for short times, since it does
essentially full convolution over the arrays.
