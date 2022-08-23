import time

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
        self.list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']#, 'R']
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
        self.general_report = pd.DataFrame(columns=list(self.list_of_muscles))

        self.create_dataset_across_speeds()
        # print("Data prep done")
        # self.train_model_with_all_channels()

    def get_speed_specific_profile_per_subject(self):
        columns = ['Speed', 'Training RMSE', 'Validation RMSE', 'Training accuracy', 'Validation accuracy', 'Epochs ran']
        speed_performance_profiler = pd.DataFrame(columns=columns)
        for speed in self.list_of_speeds:
            speed = self.list_of_speeds[8]
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, speed, self.list_of_muscles,
                                                                      label_name=self.label_name)
            prepped_data = CNNLSTMDataPrep(EMG_signals, labels, window_length=512, window_step=40,
                                           batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                           lstm_sequences=False, split_data=True, shuffle_full_dataset=False)
            x_train, y_train, x_test, y_test = prepped_data.prepped_data
            RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                  saved_model_name=self.subject + '_all_muscles_at_speed_' + speed,
                                  angle_range=90, load_model=False)
            RunningModel.train_network()
            new_row = pd.DataFrame([[speed, RunningModel.recorded_training_error, RunningModel.recorded_validation_error,
                                    RunningModel.recorded_training_error/(torch.max(y_train).item() - torch.min(y_train).item()),
                                    RunningModel.recorded_validation_error/(torch.max(y_test).item() - torch.min(y_test).item()),
                                    RunningModel.epochs_ran]], columns=columns)
            speed_performance_profiler = pd.concat([speed_performance_profiler, new_row], ignore_index=True)
        speed_performance_profiler.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject + '_all_speeds.csv')

    def get_pace_profile_per_subject(self):
        columns = ['Pace', 'Training RMSE', 'Validation RMSE', 'Training accuracy', 'Validation accuracy',
                   'Epochs ran']
        paces = ['Slow', 'Medium', 'Fast']
        speed_performance_profiler = pd.DataFrame(columns=columns)
        pace_idx = 0
        for pace in range(0, len(self.list_of_speeds), 4):
            training_signals = []
            training_labels = []
            for speed in range(4):
                EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, self.list_of_speeds[pace+speed],
                                                                          self.list_of_muscles,
                                                                          label_name=self.label_name)
                training_signals.append(EMG_signals)
                training_labels.append(labels)
            training_signals = np.concatenate(training_signals, axis=0)
            training_labels = np.concatenate(training_labels, axis=0)
            prepped_data = CNNLSTMDataPrep(training_signals, training_labels, window_length=512, window_step=40,
                                           batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                           lstm_sequences=False, split_data=True, shuffle_full_dataset=True)
            x_train, y_train, x_test, y_test = prepped_data.prepped_data
            RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                  saved_model_name=self.subject + '_all_muscles_at_pace_' + paces[pace_idx],
                                  angle_range=90, load_model=False)
            RunningModel.train_network()
            new_row = pd.DataFrame(
                [[paces[pace_idx], RunningModel.recorded_training_error, RunningModel.recorded_validation_error,
                  RunningModel.recorded_training_error / (torch.max(y_train).item() - torch.min(y_train).item()),
                  RunningModel.recorded_validation_error / (torch.max(y_test).item() - torch.min(y_test).item()),
                  RunningModel.epochs_ran]], columns=columns)
            speed_performance_profiler = pd.concat([speed_performance_profiler, new_row], ignore_index=True)
            pace_idx += 1
        speed_performance_profiler.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' +
                                          self.subject + '_paces.csv')

    def create_dataset_across_speeds(self):
        training_signals = []
        training_labels = []
        reserved_signals = []
        reserved_labels = []
        for i in range(len(self.list_of_speeds)-1):
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, self.list_of_speeds[i],
                                                                      self.list_of_muscles, label_name=self.label_name)
            cut_off_idx = int(EMG_signals.shape[0] * (1.0 - self.reserved_fraction))
            training_signals.append(EMG_signals[0:cut_off_idx, :])
            training_labels.append(labels[0:cut_off_idx, :])
            reserved_signals.append(EMG_signals[cut_off_idx::, :])
            reserved_labels.append(labels[cut_off_idx::, :])
        self.training_signals = np.concatenate(training_signals, axis=0)
        self.training_labels = np.concatenate(training_labels, axis=0)
        self.reserved_signals = np.concatenate(reserved_signals, axis=0)
        self.reserved_labels = np.concatenate(reserved_labels, axis=0)

    def train_model_with_all_channels(self):
        prepped_data = CNNLSTMDataPrep(self.training_signals, self.training_labels, window_length=512, window_step=40,
                                       batch_size=128, sequence_length=15, label_delay=0, training_size=0.9,
                                       lstm_sequences=False, split_data=True, shuffle_full_dataset=True)
        x_train = prepped_data.x_train
        y_train = prepped_data.y_train
        y_test = prepped_data.y_test
        x_test = prepped_data.x_test
        RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=1,
                              saved_model_name='TCN_all_channels_' + self.subject, angle_range=90)
        RunningModel.train_network()
        self.performance_report = pd.DataFrame({'Number of channels': 15, 'Number of epochs': RunningModel.epochs_ran,
                                                'Training loss': RunningModel.recorded_training_error,
                                                'Validation loss': RunningModel.recorded_validation_error,
                                                'Training accuracy': 1 - (RunningModel.recorded_training_error /
                                                                          (torch.max(y_train).item() -
                                                                           torch.min(y_train).item())),
                                                'Validation accuracy': 1 - (RunningModel.recorded_validation_error /
                                                                            (torch.max(y_test).item() -
                                                                             torch.min(y_test).item())),
                                                'Electrode removed': 'None'}, index=[0])

    def train_with_one_drop_out(self):
        self.train_model_with_all_channels()
        self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                                       '_channel_selection_performance.csv')
        self.updated_training_signals = self.training_signals.copy()
        counter = self.training_signals.shape[-1]
        while counter > 4:
            training_rmse = []
            validation_rmse = []
            epochs_ran = []
            for i in range(len(self.list_of_muscles)):
                # set the column to 0 to drop-out one of the electrodes
                if np.count_nonzero(self.updated_training_signals[:, i]) == 0:
                    training_rmse.append(999)
                    validation_rmse.append(999)
                    epochs_ran.append(999)
                else:
                    train_signals = self.updated_training_signals.copy()
                    print(self.updated_training_signals[0, :])
                    train_signals[:, i] = 0
                    print("The training signals for this loop are ", train_signals[0, :])
                    prepped_data = CNNLSTMDataPrep(train_signals, self.training_labels,
                                                   window_length=512, window_step=40, batch_size=128, sequence_length=15,
                                                   label_delay=0, training_size=0.9, lstm_sequences=False,
                                                   split_data=True, shuffle_full_dataset=True)
                    x_train, y_train, x_test, y_test = prepped_data.prepped_data
                    RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=80,
                                          saved_model_name='TCN_' + self.subject + '_' + str(counter-1) +
                                                           '_electrodes_without_' + self.list_of_muscles[i],
                                          angle_range=90)
                    RunningModel.train_network()
                    training_rmse.append(RunningModel.recorded_training_error)
                    validation_rmse.append(RunningModel.recorded_validation_error)
                    epochs_ran.append(RunningModel.epochs_ran)
                    print(training_rmse)
                    print(validation_rmse)
                    print(epochs_ran)
                    print("WE HAVE JUST FINISHED LOOP NUMBER ", i)
            new_row = pd.DataFrame([training_rmse], columns=self.general_report.columns, index=['Training error with ' + str(counter - 1) + ' electrodes'])
            self.general_report = pd.concat([self.general_report, new_row])
            new_row = pd.DataFrame([validation_rmse], columns=self.general_report.columns,
                                   index=['Validation error with ' + str(counter - 1) + ' electrodes'])
            self.general_report = pd.concat([self.general_report, new_row])
            new_row = pd.DataFrame([epochs_ran], columns=self.general_report.columns,
                                   index=['Epochs ran with ' + str(counter - 1) + ' electrodes'])
            self.general_report = pd.concat([self.general_report, new_row])
            electrode_to_remove = validation_rmse.index(min(validation_rmse))
            print("The electrode to remove is ", electrode_to_remove)
            # we want to remove the electrode whose absence has the least impact on model accuracy
            new_row = pd.DataFrame([[counter-1, epochs_ran[electrode_to_remove], training_rmse[electrode_to_remove],
                                    validation_rmse[electrode_to_remove], 1 - (training_rmse[electrode_to_remove] /
                                                                               (np.max(self.training_labels) -
                                                                                np.min(self.training_labels))),
                                    1 - (validation_rmse[electrode_to_remove] / (np.max(self.training_labels) -
                                                                                 np.min(self.training_labels))),
                                    self.list_of_muscles[electrode_to_remove]]], columns=self.performance_report.columns)
            self.performance_report = pd.concat([self.performance_report, new_row], ignore_index=True)
            self.updated_training_signals[:, electrode_to_remove] = 0
            self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                                           '_channel_selection_performance.csv')
            self.general_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                                       '_channel_info.csv')
            counter -= 1
        self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                                       '_channel_selection_performance.csv')
        self.general_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                                   '_channel_info.csv')


if __name__ == "__main__":
    SelectionProcess('DS02', label_name='LKneeAngles')

