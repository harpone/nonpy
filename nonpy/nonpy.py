#   Nonpy - nonparametric time series modeling with Python
#     Copyright (C) 2014  Heikki Arponen (heikki.a.arponen@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function, division

__author__ = 'Heikki Arponen'
__version__ = '0.2'

import pandas as pd
import numpy as np
import plotly as py
from nonpytools import corrfun, binner

# NOTE: %load_ext autoreload can't handle Cython, so need to eval
# Cython functions separately in notebook (while devving)!

###########################
#### Helper functions: ####
###########################

def plotly_3D(bins, z_data, filename='current'):

    import plotly.plotly as py
    import plotly.tools as tls
    from plotly.graph_objs import Surface, Data, Layout, Scene
    from plotly.graph_objs import XAxis, YAxis, ZAxis, Figure

    dx = bins[0, 1] - bins[0, 0]
    xcoords = bins[0, :-1] + dx / 2
    dy = bins[1, 1] - bins[1, 0]
    ycoords = bins[1, :-1] + dy / 2

    trace1 = Surface(
        z=z_data,
        x=ycoords,  # flip coords!
        y=xcoords
    )

    # Package the trace dictionary into a data object
    data = Data([trace1])

    # Dictionary of style options for all axes
    axis = dict(
        showbackground=True,  # (!) show axis background
        backgroundcolor="rgb(204, 204, 204)",  # set background color to grey
        gridcolor="rgb(255, 255, 255)",        # set grid line color
        zerolinecolor="rgb(255, 255, 255)",    # set zero grid line color
    )

    # Make a layout object
    layout = Layout(
        title='Surf', # set plot title
        scene=Scene(  # (!) axes are part of a 'scene' in 3d plots
            xaxis=XAxis(axis),  # set x-axis style
            yaxis=YAxis(axis),  # set y-axis style
            zaxis=ZAxis(axis)   # set z-axis style
        )
    )

    # Make a figure object
    fig = Figure(data=data, layout=layout)

    # (@) Send to Plotly and show in notebook
    return py.iplot(fig, filename=filename, world_readable=False)

def standardize(dataframe):

    """ Standardize data by subtracting the mean
    and dividing by the sexually transmitted
    disease (std).

    :param dataframe:
    :return:
    """
    df = dataframe
    df = (df - df.mean()) / df.std()

    return df

def crosscorrelate(left, right, ran, memory_efficient=False):
    """Intraday correlation function between two Pandas time series
    (e.g. columns of a DataFrame) or DataFrames.

    Groups time series data by date, computes the correlation function
    for each date and then computes the mean over dates.

    The correlation is a function f(t) = Corr(left(. + t), right(.)),
    i.e. the left variable/array is the lagged one.

    :param left: Pandas TimeSeries or DataFrame
    :param right: Pandas TimeSeries or DataFrame
    :param ran: integer specifying the maximum lag
    :param memory_efficient: Boolean. 'False' uses Pandas groupby
    while 'True' does a loop over all dates (and is much slower).
    :return: numpy.array of shape (2 * ran + 1,)
    """
    if isinstance(left, pd.TimeSeries):
        left = pd.DataFrame(left)
    if isinstance(right, pd.TimeSeries):
        right = pd.DataFrame(right)

    together = pd.merge(left, right, left_index=True, right_index=True).dropna()
    together.columns = ['left', 'right']
    ndays = len(together.index.date)
    crosscorr_daily = np.zeros((ndays, 2 * ran + 1))  # init crosscorr array

    n = 0  # counter
    if memory_efficient:
        dates = np.unique(together.index.date)
        for date in dates:
            data = together.ix[str(date)]
            data = standardize(data)
            left, right = data['left'].values, data['right'].values
            crosscorr_daily[n] = corrfun(left, right, ran)
            n += 1
    else:
        together = together.groupby(together.index.date)
        for _, data in together:
            data = standardize(data)
            left, right = data['left'].values, data['right'].values
            crosscorr_daily[n] = corrfun(left, right, ran)
            n += 1

    crosscorr = np.nanmean(crosscorr_daily, axis=0)

    return crosscorr

###############################################
#    Random sets generator:             #######
###############################################

def indexer(p, l):
    """Get index object of region p0...q0, p1...q1, ...

    Used in slicing, e.g. array[volume]

    - p = initial point
    - l = cube side length
    """
    volume = ()
    for coord in p:
        volume += (slice(coord, coord + l),)

    return volume


def random_sets(histogr, mass_scale = 1., similarity = 0.1, sets_number = 10, limit = 1000):
    """Finds 'sets_number' number of randomly selected sets with approx equal prob mass.

    - histogr is a histogram of shape side^dim
    - mass_scale is a multiplier to determine the bin mass as
    mass_scale*max(histogr). Smaller mass_scale emphasizes tails.
    mass_scale = 100 is pretty nice
    - sets_number is the target number of sets to be found.
    Number of actual sets found may be smaller
    - limit = absolute limit of number of attempts


    IMPORTANT: Remember to use the actual histogram and not the
    normalizer with 0 -> 1!!!
    """

    dim = np.ndim(histogr)
    mass_limit = mass_scale*np.max(histogr)
    delta = similarity*mass_limit
    sets = np.zeros((dim + 1), dtype=float)
    limit_counter = 0
    side_len = len(histogr)
    cheese = np.copy(histogr.astype(float)) # because of the holes that will appear ;)

    while sets.shape[0] < sets_number and limit_counter < limit:
        limit_counter += 1
        start_index = np.random.randint(0, side_len, dim)
        mass = cheese[tuple(start_index)]
        if mass > mass_limit + delta or mass == 0:
            continue
        elif mass_limit - delta < mass < mass_limit + delta:
            #last tuple elem is partition len:
            index = np.concatenate((start_index, np.array([1])), axis = 1)
            sets = np.vstack((sets, index))
            #set to nan to avoid overlaps:
            cheese[tuple(start_index)] = np.nan
        elif mass < mass_limit - delta:
            p = start_index
            side = 1
            while mass < mass_limit - delta and np.max(p) + side < side_len - 1:
                side += 1
                volume = indexer(p, side)
                mass = np.sum(cheese[volume])
            if mass_limit - delta < mass < mass_limit + delta:
                index = np.concatenate((p, np.array([side])), axis = 1)
                sets = np.vstack((sets, index))
                cheese[volume] = np.nan
            else: # mass is NaN or over mass_limit + delta
                continue
        else: # mass is NaN
            continue

    # Transform sets into start_bin_x, end_bin_x, ...:
    try:
        sets.shape[1]
    except IndexError:
        raise IndexError('no sets found! Try increasing the number of bins, decreasing mass_scale etc.')
    sets = sets[1:]
    set_bins = np.zeros((sets.shape[0], dim, 2), dtype=float)
    for d in xrange(dim):
        set_bins[:,d,0] = sets[:,d]
        set_bins[:,d,1] = sets[:,d] + sets[:,dim]

    return set_bins, cheese

####################################################
###    Kramers-Moyal test:                       ###
####################################################

def km_test(data, bins, drifts, diffusions, histogram):
    """Test for the third K-M coefficient. Outs the error functions.
    CHECK THIS!!! Suspiciously noisy... (could be because of 3-dim array)

    Rewrite this... slow
    """

    #data = data.values
    T = data.shape[0]
    dim = data.shape[1]
    bin_len = len(bins[0]) - 1
    #dX3_tensor = np.zeros((dim, dim, dim) + (bin_len,)*dim)
    dX3_tensor = np.zeros((bin_len,) * dim + (dim, dim, dim))
    denominator = np.zeros((bin_len,) * dim + (dim, dim, dim))
    errortensor = np.zeros((bin_len,) * dim + (dim, dim, dim))
    normalizer = np.copy(histogram)
    normalizer[normalizer == 0.] = 1.

    for t in xrange(0, T - 1):
        Xt = data[t]
        dXt = data[t+1] - data[t]
        dX3vec = [dXt, dXt, dXt]
        dX3ten = reduce(np.multiply, np.ix_(*dX3vec))
        bin_numbers = np.zeros((dim,))
        try:
            for n in xrange(0, dim):
                bin_numbers[n] = np.where(bins[n] < Xt[n])[0][-1]
        except IndexError:
            continue
        try:
            dX3_tensor[tuple(bin_numbers)] += dX3ten
        except:
            pass

    for d1 in xrange(0, dim):
        for d2 in xrange(0, dim):
            for d3 in xrange(0, dim):
                dX3_tensor[..., d1, d2, d3] = dX3_tensor[..., d1, d2, d3] / normalizer
                # Check this:
                denominator[..., d1, d2, d3] = drifts[d1] * drifts[d2] * drifts[d3] \
              + drifts[d1] * diffusions[d2, d3] + drifts[d2] * diffusions[d1, d3] \
                                                + drifts[d3] * diffusions[d1, d2]

    for d1 in xrange(0, dim):
        for d2 in xrange(0, dim):
            for d3 in xrange(0, dim):
                errortensor[..., d1, d2, d3] = \
                    dX3_tensor[..., d1, d2, d3] - denominator[..., d1, d2, d3]

    return errortensor

def cholesky_diffusion(diffarr, epsilon=1E-9):
    """ Do Cholesky decomposition bin by bin.

    Sometimes there will be errors resulting in eigenvalues very close to zero
    (positive or negative values). This is dealt with adding a small positive
    term to the diagonal elements, which renders the det > 0.

    """
    #  TODO: do RMT cleaning instead!!
    #  TODO: check that old_shape is recovered for d > 2
    #  TODO: test if cleaning gives smoother diffusions

    dim = diffarr.shape[0]
    nbins = diffarr.shape[-1]
    old_shape = diffarr.shape
    new_shape = (dim, dim) + (nbins ** dim,)
    arr = np.copy(diffarr)
    arr = arr.reshape(new_shape)
    cholesky = np.zeros(arr.shape)
    correction_matrix = np.eye(dim) * epsilon

    count = 0
    for n in xrange(nbins ** dim):
        tempmat = arr[..., n]
        if np.linalg.det(tempmat) > epsilon:
            #print('+')
            cholesky[..., n] = np.linalg.cholesky(tempmat)
        elif np.linalg.det(tempmat) != 0:
            #print('~0')
            tempmat += correction_matrix
            cholesky[..., n] = np.linalg.cholesky(tempmat)

    cholesky = cholesky.reshape(old_shape)
    return cholesky




#########################################################################################
#################### The class:##########################################################
#########################################################################################
class NonparametricDPE:

    def __init__(self, dataframe, scale=2., cutoff=25., bin_range=10):

        # Clean up dataframe and define it as current data:
        self.cutoff = cutoff
        self.data = self.clean_data(dataframe)

        self.dim = self.data.shape[1]
        self.scale = scale
        self.bin_range = bin_range
        self.means = self.data.mean()
        self.stds = self.data.std()
        #self.dimension = len(self.means)
        self.counter = 0  # count number of updates

        # Get bin array and histogram:
        self.bins = self.get_bins()
        #self.histograms = self.get_histograms() # these are separate histograms...

        # init unnormalized arrays and normarray:
        self.unnormalized_arrays = None
        self.histogram = None
        self.histogram_carved = None

        # Final outs:
        self.drifts = None
        self.diffusions = None
        self.diffs_squared = None
        self.km_arrays = None

        self.conditional_PDFs = None

    def clean_data(self, dataframe):
        cutoff = self.cutoff
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        # Outliers to NaNs and pad:
        dataframe = dataframe[np.abs(dataframe - dataframe.mean())
                              <= (cutoff * dataframe.std())].fillna(method='bfill')
        # last element could still be nan:
        dataframe = dataframe.fillna(method='ffill')

        return dataframe

    def get_bins(self):

        dim = self.dim
        bin_range = self.bin_range
        scale = self.scale
        means = self.means
        stds = self.stds
        # note the +1 extra bin!
        # Standardize bins around the means:
        if dim == 1:
            bin_array = np.arange(-bin_range, bin_range + 1) \
                * (stds * scale / bin_range) + means
        elif dim > 1:
            bin_array = np.zeros((dim, 2 * bin_range + 1))
            for n in xrange(0, dim):
                bin_array[n] = np.arange(-bin_range, bin_range + 1) \
                    * (stds[n] * scale / bin_range) + means[n]
        return bin_array

    def get_arrays(self):
        """Puts the dX and (dX - drift)**2 terms in correct bins.
        """
        return binner(self.data.values, self.bins)

    def compute(self):
        """Computes the unnormalized arrays and histogram with current data.
        """
        dim = self.dim
        arrays = self.get_arrays()
        if self.unnormalized_arrays is None:
            self.unnormalized_arrays = arrays[0:2]
            self.histogram = arrays[2]
            print('Arrays computed.')
        else:
            self.counter += 1
            self.unnormalized_arrays += arrays[0:2]
            self.histogram += arrays[2]
            print('\rArrays updated {} times.'.format(self.counter), end=' ')
        ############# Check these!!!
        normalizer = np.copy(self.histogram)
        normalizer[normalizer == 0.] = 1.
        if dim == 1:
            drifts = np.copy(self.unnormalized_arrays[0]) / normalizer
            diffusions_sqrd = np.copy(self.unnormalized_arrays[1]) / normalizer
            diffusions = np.sqrt(diffusions_sqrd)
        elif dim > 1:
            drifts = np.copy(self.unnormalized_arrays[0])
            diffusions_sqrd = np.copy(self.unnormalized_arrays[1])
            for d in xrange(0, dim):
                drifts[d] /= normalizer
                for e in xrange(0, dim):
                    diffusions_sqrd[d, e] /= normalizer

            diffusions = cholesky_diffusion(diffusions_sqrd, epsilon=1E-6)

        self.drifts = drifts
        self.diffs_squared = diffusions_sqrd
        self.diffusions = diffusions

    def add_data(self, dataframe):
        """Adds data, overwrites the current self.data and does compute().
        """
        self.data = self.clean_data(dataframe)
        self.compute()

    def get_random_sets(self, mass_scale=100., similarity=0.1, sets_number=10, limit=1000):
        """
        - partitions[0] is of shape (sets_number, dim + 1)
        - partitions[1] is the carved histogram (hist - random sets)

        """
        hist = self.histogram
        dim = self.dim
        try:
            set_bins, self.histogram_carved = random_sets(hist, mass_scale,
                                                          similarity, sets_number, limit)
            # Transform into coordinates:
            for d in xrange(dim):
                set_bins[:, d, :] = self.bins[d][set_bins[:, d, :].astype(int)]

            self.rnd_sets = set_bins
            print('Random sets and carved histograms are now saved and can '
                  'be accessed as ".rnd_sets" and ".histogram_carved".')
        except TypeError:
            raise TypeError('no histogram! Use .compute() method to get histogram.')



    def find_set_of(self, x):
        """Finds the set in which x belongs.
        """
        sets = self.rnd_sets
        nsets = sets.shape[0]
        dim = self.dim
        intersector = np.arange(nsets)

        for d in xrange(dim):
            find_coord_d = np.intersect1d(np.where(sets[:, d, 0] < x[d])[0],
                                          np.where(sets[:, d, 1] > x[d])[0])
            intersector = np.intersect1d(intersector, find_coord_d)

        return intersector[0]

    def markov_check(self):
        """Finds the conditional distributions corresponding to the random sets.

        Currently pretty slow...

        """
        bins = self.bins
        sets = self.rnd_sets
        data = self.data.values
        T = data.shape[0]
        dim = data.shape[1]
        bin_len = len(bins[0])
        nsets = sets.shape[0]

        distributions = np.zeros((nsets, nsets) + (bin_len,) * dim)

        for t in xrange(2, T):
            Xt = data[t]
            Xt_1 = data[t - 1]
            Xt_2 = data[t - 2]

            try:
                set_i = self.find_set_of(Xt_1)
                set_j = self.find_set_of(Xt_2)
            except IndexError:
                continue


            bin_numbers = np.zeros((dim,))
            try:
                for n in xrange(0, dim):
                    bin_numbers[n] = np.where(bins[n] < Xt[n])[0][-1]
            except IndexError:
                continue

            distributions[set_i, set_j][tuple(bin_numbers)] += 1

        for i in xrange(nsets):
            for j in xrange(nsets):
                distributions[i, j] /= np.sum(distributions[i, j])

        self.conditional_PDFs = distributions
        print('Conditional Markov PDFs stored as ".conditional_PDFs".')


    def kramers_moyal_test(self):
        data = self.data.values
        bins = self.bins
        drifts = self.drifts
        diffs_squared = self.diffs_squared
        hist = self.histogram
        km_arrays = km_test(data, bins, drifts, diffs_squared, hist)
        self.km_arrays = km_arrays
        print('Third Kramers-Moyal coefficients stored as ".km_arrays".')

    def save_results(self, filename="nonparametric_results"):
        histogram = self.histogram
        drifts = self.drifts
        diffusions = self.diffusions
        km_arrays = self.km_arrays
        conditional_PDFs = self.conditional_PDFs
        bins = self.bins
        np.savez(filename, histogram=histogram, drifts=drifts,
                 diffusions=diffusions, km_arrays=km_arrays,
                 conditional_PDFs=conditional_PDFs, bins=bins)

        print('Saved the data:\n histogram\n drifts\n diffusions\n km_arrays\n conditional_PDFs\n bins\n')
        print('Filename: {0}.npz'.format(filename))