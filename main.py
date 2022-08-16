import numpy as np
import c3d
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utility.dataprocessing import extract_c3d_data_to_hdf5, extract_hdf5_data_to_EMG_and_labels
from utility.plots import plot_the_predictions
from utility.conversions import normalise_signals
from networks import CNNLSTMDataPrep, RunConvLSTM, RunTCN
from selectionprocess import SelectionProcess

list_of_subjects = ['DS02', 'DS04', 'DS05', 'DS06', 'DS07']
list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM", "GL"]
list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])

subject_idx = 0
speed_idx = 0
label_idx = 1

training_signals = []
training_labels = []
reserved_signals = []
reserved_labels = []
for i in range(len(list_of_speeds)-1):
    EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(list_of_subjects[subject_idx], list_of_speeds[i],
                                                              list_of_muscles, label_name=list_angles[label_idx])
    cut_off_idx = int(EMG_signals.shape[0] * 0.95)
    training_signals.append(normalise_signals(EMG_signals[0:cut_off_idx, :]))
    training_labels.append(labels[0:cut_off_idx, :])
    reserved_signals.append(normalise_signals(EMG_signals[cut_off_idx::, :]))
    reserved_labels.append(labels[cut_off_idx::, :])
training_signals = np.concatenate(training_signals, axis=0)
training_labels = np.concatenate(training_labels, axis=0)
reserved_signals = np.concatenate(reserved_signals, axis=0)
reserved_labels = np.concatenate(reserved_labels, axis=0)


# SelectionProcess(list_of_subjects[subject_idx], list_angles[label_idx], reserved_fraction=0.05)
# exit()

EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(list_of_subjects[subject_idx], list_of_speeds[speed_idx],
                                                          list_of_muscles, label_name=list_angles[label_idx])

# training_signals = training_signals[0:EMG_signals.shape[0], :]
# training_labels = training_labels[0:EMG_signals.shape[0], :]

x_train, x_test, y_train, y_test = CNNLSTMDataPrep(training_signals, training_labels, window_length=512, window_step=40,
                                                   batch_size=64, sequence_length=15, label_delay=0,
                                                   training_size=0.85, lstm_sequences=False, split_data=True,
                                                   shuffle_full_dataset=False).prepped_data


x_train = x_train[:, :, :, 0:100]
y_train = y_train[:, :, 0:100]

RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=80, saved_model_name='TCN_Test', angle_range=90)
# RunningModel = RunConvLSTM(x_train, y_train, x_test, y_test, n_channels=15, lstm_hidden=64, epochs=80,
#                            saved_model_name='ConvLSTM_Test', lstm_layers=2)
RunningModel.train_network()
print("Number of epochs:", RunningModel.epochs_ran)
print("Final training RMSE: ", RunningModel.recorded_training_error)
print("Final training accuracy: ", 1.0 - (RunningModel.recorded_training_error/(torch.max(y_train).item()-torch.min(y_train).item())))
print("Final validation RMSE: ", RunningModel.recorded_validation_error)
print("Final validation accuracy: ", 1.0 - (RunningModel.recorded_validation_error/(torch.max(y_test).item()-torch.min(y_test).item())))
plot_the_predictions(RunningModel.model, RunningModel.saved_model_path, RunningModel.saved_model_name, x_test, y_test,
                     lstm_hidden_size=64, lstm_layers=0)



