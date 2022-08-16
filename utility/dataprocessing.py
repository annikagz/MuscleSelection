import math

import numpy as np
import c3d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utility.conversions import upsample_based_on_axis


def extract_c3d_data_to_hdf5(list_of_subjects, list_of_speeds):
    list_emg = list(["Sensor {}.IM EMG{}".format(i, i) for i in range(1, 9)]) + \
               list(["Sensor {}.IM EMG{}".format(i, i) for i in range(10, 17)])
    list_muscles = list(
        ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM",
         "GL"])
    list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])
    for subject in list_of_subjects:
        for speed in list_of_speeds:
            filename = subject + '_' + speed
            reader = c3d.Reader(open('/media/ag6016/Storage/MuscleSelection/c3d files/' + filename + '.c3d', 'rb'))
            required_analog_idx = [i for i in range(len(reader.analog_labels)) if reader.analog_labels[i].strip() in list_emg]
            required_point_idx = [i for i in range(len(reader.point_labels)) if reader.point_labels[i].strip() in list_angles]
            stored_EMG_data = []  # np.expand_dims(np.array(list_emg), axis=0)
            stored_kinematics_data = []  # np.expand_dims(np.array(list_angles), axis=0)
            for frame_no, points, analog in reader.read_frames():
                emg_data = np.array([analog[i, :] for i in required_analog_idx]).T
                point_data = np.expand_dims(np.array([points[i, 0] for i in required_point_idx]), axis=0)
                stored_EMG_data.append(emg_data)  # = np.concatenate((stored_EMG_data, emg_data), axis=0)
                stored_kinematics_data.append(point_data)  # = np.concatenate((stored_kinematics_data, point_data), axis=0)
            stored_EMG_data = np.concatenate(stored_EMG_data, axis=0)
            stored_kinematics_data = np.concatenate(stored_kinematics_data, axis=0)
            stored_kinematics_data = upsample_based_on_axis(stored_kinematics_data, stored_EMG_data.shape[0], axis=0)
            full_data = pd.DataFrame(np.concatenate((stored_EMG_data, stored_kinematics_data), axis=1),
                                     columns=list_muscles + list_angles)
            full_data.to_hdf('/media/ag6016/Storage/MuscleSelection/full_data/' + filename + '.h5', key='Data')
            print(filename, " SAVED")


def extract_hdf5_data_to_EMG_and_labels(subject, speed, list_of_muscles, label_name):
    data = pd.read_hdf('/media/ag6016/Storage/MuscleSelection/full_data/' + subject + '_' + speed + '.h5', key='Data')
    EMG_signals = data[list_of_muscles].to_numpy()
    label = np.expand_dims(data[label_name].to_numpy(), axis=1)
    return EMG_signals, label


def split_signals_into_TCN_windows(signals, labels, window_length, window_step, label_delay=0, reps_first=True):
    signal_snippets = []
    label_snippets = []
    for i in range(0, int(signals.shape[0] - window_length), window_step):
        signal_snippets.append(np.expand_dims(signals[i: i + window_length, :], axis=2))
        label_snippets.append(labels[i + 1 + label_delay, :])
    windowed_signals = np.concatenate(signal_snippets, axis=2)#.transpose((2, 1, 0))
    windowed_labels = np.array(label_snippets).transpose((1, 0))
    if reps_first:
        windowed_signals = windowed_signals.transpose((2, 0, 1))
        windowed_labels = windowed_labels.transpose((1, 0))
    return windowed_signals, windowed_labels


def group_windows_into_sequences(signals, labels, n_windows_per_sequence, window_axis=-1):
    cropped_n_windows = math.floor(signals.shape[window_axis] / n_windows_per_sequence)
    signals = signals[:, :, 0: cropped_n_windows*n_windows_per_sequence]
    signals = signals.reshape((signals.shape[0], signals.shape[1], n_windows_per_sequence, -1), order='F').transpose((2, 0, 1, 3))
    labels = labels[:, 0: cropped_n_windows*n_windows_per_sequence].reshape((n_windows_per_sequence, labels.shape[0], -1), order='F')
    return signals, labels


def shuffle(signals, labels):
    shuffler = np.random.permutation(signals.shape[-1])
    if signals.ndim == 4:
        signals = signals[:, :, :, shuffler]
        labels = labels[:, :, shuffler]
    elif signals.ndim == 3:
        signals = signals[:, :, shuffler]
        labels = labels[:, shuffler]
    elif signals.ndim == 5:
        signals = signals[:, :, :, :, shuffler]
        labels = labels[:, :, :, shuffler]
    return signals, labels


def split_into_train_test(signals, labels, train_size, split_axis):
    if signals.ndim == 4:
        if split_axis == -1:
            signals = signals.transpose((3, 0, 1, 2))
            labels = labels.transpose((2, 0, 1))
        elif split_axis != -1 and split_axis!= 0:
            raise ValueError('You need to have the split axis be either in first or last place.')
        x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=train_size, shuffle=False)
        x_train = x_train.transpose((1, 2, 3, 0))
        x_test = x_test.transpose((1, 2, 3, 0))
        y_train = y_train.transpose((1, 2, 0))
        y_test = y_test.transpose((1, 2, 0))
    elif signals.ndim == 3:
        if split_axis == -1:
            signals = signals.transpose((2, 0, 1))
            labels = labels.transpose((1, 0))
        elif split_axis != -1 and split_axis!= 0:
            raise ValueError('You need to have the split axis be either in first or last place.')
        x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=train_size, shuffle=False)
        x_train = x_train.transpose((2, 1, 0))
        x_test = x_test.transpose((2, 1, 0))
        y_train = y_train.transpose((1, 0))
        y_test = y_test.transpose((1, 0))
    else:
        raise ValueError('Your input arrays have the wrong shape.')
    return x_train, x_test, y_train, y_test


def split_into_batches(x_train, y_train, x_test, y_test, batch_size, batch_axis=1):
    if x_train.ndim == 4:
        cropped_length = int(x_train.shape[-1] - (x_train.shape[-1] % batch_size))
        x_train = x_train[:, :, :, :cropped_length]
        y_train = y_train[:, :, :cropped_length]
        x_batches = []
        for i in range(0, int(x_train.shape[-1]), int(batch_size)):
            x_batches.append(np.expand_dims(x_train[:, :, :, i:i+batch_size], axis=3))
        x_train = np.concatenate(x_batches, axis=3)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], -1, batch_size))
        x_transposition = [0, 1, 2, 3]
        x_transposition.insert(int(batch_axis), 4)
        y_transposition = [0, 1, 2]
        y_transposition.insert(int(batch_axis), 3)
        x_train = x_train.transpose(x_transposition)
        y_train = y_train.transpose(y_transposition)
        x_test = np.expand_dims(x_test, axis=batch_axis)
        y_test = np.expand_dims(y_test, axis=batch_axis)
    elif x_train.ndim == 3:
        cropped_length = int(x_train.shape[-1] - (x_train.shape[-1] % batch_size))
        x_train = x_train[:, :, :cropped_length]
        y_train = y_train[:, :cropped_length]
        x_batches = []
        for i in range(0, int(x_train.shape[-1]), int(batch_size)):
            x_batches.append(np.expand_dims(x_train[:, :, i:i + batch_size], axis=2))
        x_train = np.concatenate(x_batches, axis=2)
        y_train = y_train.reshape((y_train.shape[0], -1, batch_size))
        x_transposition = [0, 1, 2]
        x_transposition.insert(int(batch_axis), 3)
        y_transposition = [0, 1]
        y_transposition.insert(int(batch_axis), 2)
        x_train = x_train.transpose(x_transposition)
        y_train = y_train.transpose(y_transposition)
        x_test = np.expand_dims(x_test, axis=batch_axis)
        y_test = np.expand_dims(y_test, axis=batch_axis)
    else:
        raise ValueError('The input dimensions of the array are wrong')
    return x_train, y_train, x_test, y_test

