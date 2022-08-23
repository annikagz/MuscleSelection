from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal
from statistics import median
from scipy.interpolate import interp1d
import torch
import math
import pandas as pd
# from utility.dataprocessing import extract_hdf5_data_to_EMG_and_labels
# from networks import CNNLSTMDataPrep


def upsample_based_on_axis(time_series, n_samples, axis=0):
    if time_series.shape[axis] != n_samples:
        xnew = np.linspace(0, time_series.shape[axis]-1, num=n_samples)
        xold = np.arange(0, time_series.shape[0])
        upsampled_series = interp1d(xold, time_series, axis=axis)(xnew)
        return upsampled_series


def normalise_signals(x_train, x_test):
    for channel in range(x_train.shape[-1]):
        x_train[:, channel] = (x_train[:, channel] - np.mean(x_train[:, channel]))
        x_test[:, channel] = (x_test[:, channel] - np.mean(x_train[:, channel]))
        x_train[:, channel] = x_train[:, channel] / (0.95 * np.max(np.abs(x_train[:, channel])))
        x_test[:, channel] = x_test[:, channel] / (0.95 * np.max(np.abs(x_train[:, channel])))
    return x_train, x_test


# def save_prepped_data(list_of_subjects, list_of_speeds, list_of_muscles, list_of_labels):
#     for subject_idx in range(len(list_of_subjects)):
#         training_signals = []
#         training_labels = []
#         reserved_signals = []
#         reserved_labels = []
#         for speed in list_of_speeds:
#             EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(list_of_subjects[subject_idx], speed,
#                                                                       list_of_muscles,
#                                                                       label_name=list_of_labels[subject_idx])
#             cut_off_idx = int(EMG_signals.shape[0] * 0.95)
#             training_signals.append(EMG_signals[0:cut_off_idx, :])
#             training_labels.append(labels[0:cut_off_idx, :])
#             reserved_signals.append(EMG_signals[cut_off_idx::, :])
#             reserved_labels.append(labels[cut_off_idx::, :])
#         training_signals = np.concatenate(training_signals, axis=0)
#         training_labels = np.concatenate(training_labels, axis=0)
#         reserved_signals = np.concatenate(reserved_signals, axis=0)
#         reserved_labels = np.concatenate(reserved_labels, axis=0)
#
#         prepped_data = CNNLSTMDataPrep(training_signals, training_labels, window_length=512, window_step=40,
#                                        batch_size=64, sequence_length=15, label_delay=0, training_size=0.85,
#                                        lstm_sequences=False, split_data=True)
#
#         x_train = prepped_data.x_train
#         y_train = prepped_data.y_train
#         x_test = prepped_data.x_test
#         y_test = prepped_data.y_test
#
#         torch.save(x_train,
#                    '/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[subject_idx] + 'x_train.pt')
#         torch.save(y_train,
#                    '/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[subject_idx] + 'y_train.pt')
#         torch.save(x_test,
#                    '/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[subject_idx] + 'x_test.pt')
#         torch.save(y_test,
#                    '/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[subject_idx] + 'y_test.pt')
#
#         print("Training data saved for ", list_of_subjects[subject_idx])
#
#         np.save('/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[
#             subject_idx] + 'reserved_signals.npy',
#                 reserved_signals)
#         np.save('/media/ag6016/Storage/MuscleSelection/prepped_data/' + list_of_subjects[
#             subject_idx] + 'reserved_labels.npy',
#                 reserved_labels)
#
#         print("Reserved data saved for ", list_of_subjects[subject_idx])
