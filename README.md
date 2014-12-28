nonpy
=====

Nonparametric time series modeling.

Fit multivariate data to a diffusion process.

Pre-alpha stage.

Currently the only useful functions are the (Cython
optimized) 'crosscorrelate' function, which does cross-
correlation (with lag) between two Pandas time series,
and 'binner' which estimates the drift and diffusion
terms of a (multivariate) diffusion process (also
Cythonized).

The reason I wrote the 'crosscorrelate' function is
that scipy.crosscorrelate does a full convolution
over the entire dataset (which could be a time series
of e.g. 1bn samples), when in practical situations one
needs only maybe 100 lag correlation. So this function
is therefore a LOT faster!

More stuff coming up soon!
