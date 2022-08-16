import torch.nn as nn
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utility.dataprocessing import split_signals_into_TCN_windows, group_windows_into_sequences, shuffle, \
    split_into_train_test, split_into_batches, extract_hdf5_data_to_EMG_and_labels
from utility.plots import plot_the_predictions
from utility.conversions import normalise_signals
from networks import RunConvLSTM, CNNLSTMDataPrep, RunTCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelectionProcess:
    """
    The point of this algorithm is to train the network and then implement a one-out process to try and remove the electrodes one by one until there are only 4 left.
    To do this, we must:
    1) Record the overall accuracy of the network with all the electrodes
    2) Have a list of the electrodes names
    3) Remove all the electrodes one by one and retrain the network each time with the remaining electrodes
    4) Compare the accuracy and remove the electrode which has the least impact on accuracy (set the channel to 0)
    5) Repeat the process until only 4 electrodes are left
    6) Retain the order in which the electrodes were removed
    6) Repeat the process until all the subjects have been done
    """
    def __init__(self, subject, label_name, reserved_fraction=0.05):
        self.list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
        self.list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA",
                                "SO", "GM", "GL"]
        self.list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])
        self.subject = subject
        self.label_name = label_name
        self.reserved_fraction = reserved_fraction
        self.training_signals = None
        self.updated_training_signals = None
        self.training_labels = None
        self.reserved_signals = None
        self.reserved_labels = None
        self.performance_report = None
        self.general_report = pd.DataFrame(columns=list(['Number of electrodes']) + list(self.list_of_muscles))

        self.create_dataset_across_speeds()
        print("Data prep done")
        self.train_model_with_all_channels()

    def create_dataset_across_speeds(self):
        training_signals = []
        training_labels = []
        reserved_signals = []
        reserved_labels = []
        for i in range(len(self.list_of_speeds)-1):
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, self.list_of_speeds[i],
                                                                      self.list_of_muscles, label_name=self.label_name)
            cut_off_idx = int(EMG_signals.shape[0] * (1.0 - self.reserved_fraction))
            training_signals.append(normalise_signals(EMG_signals[0:cut_off_idx, :]))
            training_labels.append(labels[0:cut_off_idx, :])
            reserved_signals.append(normalise_signals(EMG_signals[cut_off_idx::, :]))
            reserved_labels.append(labels[cut_off_idx::, :])
        self.training_signals = np.concatenate(training_signals, axis=0)
        self.training_labels = np.concatenate(training_labels, axis=0)
        self.reserved_signals = np.concatenate(reserved_signals, axis=0)
        self.reserved_labels = np.concatenate(reserved_labels, axis=0)

    def train_model_with_all_channels(self):
        #torch.cuda.empty_cache()
        x_train, x_test, y_train, y_test = CNNLSTMDataPrep(self.training_signals, self.training_labels,
                                                           window_length=512, window_step=40, batch_size=64,
                                                           sequence_length=15, label_delay=0, training_size=0.85,
                                                           lstm_sequences=False, split_data=True,
                                                           shuffle_full_dataset=False).prepped_data
        RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=10,
                              saved_model_name='TCN_' + self.subject, angle_range=90)
        RunningModel.train_network()
        plot_the_predictions(RunningModel.model, RunningModel.saved_model_path, RunningModel.saved_model_name, x_test,
                             y_test)
        self.performance_report = pd.DataFrame({'Number of channels': 15, 'Number of epochs': RunningModel.epochs_ran,
                                                'Training loss': RunningModel.recorded_training_error,
                                                'Validation loss': RunningModel.recorded_validation_error,
                                                'Model accuracy': RunningModel.recorded_validation_error /
                                                                  (max(self.training_labels)-min(self.training_labels)),
                                                'Electrode removed': 'None'})
        print(self.performance_report)
        exit()

    def train_with_one_drop_out(self):
        self.updated_training_signals = self.training_signals
        counter = self.training_signals.shape[-1]
        while counter > 4:
            training_values = []
            validation_values = []
            epochs_ran = []
            for i in range(len(self.list_of_muscles)):
                # set the column to 0 to drop-out one of the electrodes
                if torch.count_nonzero(self.updated_training_signals[:, i])[0] == 0:
                    training_values.append(np.nan)
                    validation_values.append(np.nan)
                    epochs_ran.append(np.nan)
                else:
                    training_signals = self.updated_training_signals
                    for val in range(training_signals.shape[0]):
                        training_signals[val, i] = 0
                    x_train, x_test, y_train, y_test = CNNLSTMDataPrep(self.updated_training_signals,
                                                                       self.training_labels, window_length=512,
                                                                       window_step=40, batch_size=64,
                                                                       sequence_length=15, label_delay=0,
                                                                       training_size=0.85, lstm_sequences=False,
                                                                       split_data=True,
                                                                       shuffle_full_dataset=False).prepped_data
                    RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=1,
                                          saved_model_name='TCN_' + self.subject + str(counter-1) +
                                                           'electrodes_without_' + self.list_of_muscles[i],
                                          angle_range=90)
                    RunningModel.train_network()
                    plot_the_predictions(RunningModel.model, RunningModel.saved_model_path,
                                         RunningModel.saved_model_name, x_test, y_test, lstm_layers=0)
                    training_values.append(RunningModel.recorded_training_error)
                    validation_values.append(RunningModel.recorded_validation_error)
                    epochs_ran.append(RunningModel.epochs_ran)
            self.general_report.append(pd.DataFrame(training_values,
                                                    index=['Training error with ' + str(counter-1) + 'electrodes'],
                                                    columns=self.general_report.columns))
            self.general_report.append(pd.DataFrame(validation_values,
                                                    index=['Validation error with ' + str(counter - 1) + 'electrodes'],
                                                    columns=self.general_report.columns))
            self.general_report.append(pd.DataFrame(epochs_ran,
                                                    index=['Epochs ran with ' + str(counter - 1) + 'electrodes'],
                                                    columns=self.general_report.columns))

            electrode_to_remove = validation_values.index(min(validation_values))
            # we want to remove the electrode whose absence has the least impact on model accuracy
            self.performance_report.append(pd.DataFrame([counter-1, epochs_ran, training_values[electrode_to_remove],
                                                         validation_values[electrode_to_remove],
                                                         validation_values[electrode_to_remove] /
                                                         (max(self.training_labels) - min(self.training_labels))],
                                                        columns=self.performance_report.columns))
            for row in range(self.updated_training_signals.shape[0]):
                self.updated_training_signals[row, electrode_to_remove] = 0
            counter -= 1




