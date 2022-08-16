from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal
from statistics import median
from scipy.interpolate import interp1d
import torch
import math
import pandas as pd


def upsample_based_on_axis(time_series, n_samples, axis=0):
    if time_series.shape[axis] != n_samples:
        xnew = np.linspace(0, time_series.shape[axis]-1, num=n_samples)
        xold = np.arange(0, time_series.shape[0])
        upsampled_series = interp1d(xold, time_series, axis=axis)(xnew)
        return upsampled_series


def normalise_signals(EMG_signals):
    for channel in range(EMG_signals.shape[-1]):
        EMG_signals[:, channel] = (EMG_signals[:, channel] - EMG_signals[:, channel].mean(axis=0))
        EMG_signals[:, channel] = EMG_signals[:, channel] / (0.95 * np.max(np.abs(EMG_signals[:, channel])))
    return EMG_signals
