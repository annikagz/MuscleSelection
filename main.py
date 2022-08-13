import numpy as np
import c3d
import pandas as pd
import matplotlib.pyplot as plt
from utility.dataprocessing import extract_c3d_data_to_hdf5, extract_hdf5_data_to_EMG_and_labels
from utility.plots import plot_the_predictions
from networks import CNNLSTMDataPrep, RunConvLSTM

list_of_subjects = ['DS02', 'DS04', 'DS05', 'DS06', 'DS07']
list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM", "GL"]
list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])

subject_idx = 0
speed_idx = 0
label_idx = 1

EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(list_of_subjects[subject_idx], list_of_speeds[speed_idx],
                                                          list_of_muscles, label_name=list_angles[label_idx])

x_train, x_test, y_train, y_test = CNNLSTMDataPrep(EMG_signals, labels, window_length=512, window_step=30,
                                                   batch_size=16, sequence_length=15, label_delay=0,
                                                   training_size=0.95, shuffle_full_dataset=False).prepped_data

RunningModel = RunConvLSTM(x_train, y_train, x_test, y_test, n_channels=15, lstm_hidden=16, epochs=80, saved_model_name='ConvLSTM_Test', lstm_layers=2)
RunningModel.train_network()
print("Number of epochs:", RunningModel.epochs_ran)
print("Final training error: ", RunningModel.recorded_training_error)
print("Final validation error: ", RunningModel.recorded_validation_error)
plot_the_predictions(RunningModel.model, RunningModel.saved_model_path, RunningModel.saved_model_name, x_test, y_test)


