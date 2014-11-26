__author__ = 'Heikki Arponen'
version = '0.16'

import pandas as pd
import numpy as np

from nonpytools import corrfun
# NOTE: %load_ext autoreload can't handle Cython, so need to import
# Cython modules separately in notebook (while devving)!#
#

def standardize(dataframe):

    df = dataframe
    df = (df - df.mean()) / df.std()

    return df

def crosscorrelate(left, right, ran):
    """Intraday correlation function between two Pandas time series
    (e.g. columns of a DataFrame).

    Groups time series data by date, computes the correlation function
    for each date and then computes the mean over dates.

    :param left:
    :param right:
    :param ran:
    :return:
    """
    combined_df = pd.concat([left, right], axis=1).fillna(method='bfill').fillna(method='ffill')
    combined_df.columns = ['left', 'right']
    T = len(combined_df.index.date)
    combined_df = combined_df.groupby(combined_df.index.date)
    crosscorr_daily = np.zeros((T, 2 * ran + 1))

    n = 0
    for _, data in combined_df:
        data = standardize(data)
        left, right = data['left'].values, data['right'].values
        crosscorr_daily[n] = corrfun(left, right, ran)
        n += 1

    crosscor = np.nanmean(crosscorr_daily, axis=0)

    return crosscor