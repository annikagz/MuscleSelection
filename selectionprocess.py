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
from networks import RunConvLSTM, CNNLSTMDataPrep, RunTCN, RunMLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PCAEvaluation:
    """
    For this class, we want to evaluate the muscle groups that were selected by the PCA and then save the values into a
    dataframe
    1) load the dataframe that contains the muscle groups, depending on whether we are looking at the max or the min
    variance
    2) Evaluate the performance of these muscle groups across different conditions:
        a) Train on all steady-state speeds and test on transient
        b) Train on faster speeds and test on slow speeds
        c) Train on faster speeds and test on medium speeds
    3) Record the value of all of these into some data frames under the folder Variance reports, recording both the
    training RMSE and accuracy
    """
    def __init__(self, initial_lr, training_speeds, testing_speeds, batch_size, report_name, epochs, PCA_selection_used='max', reduce_testing_set=None):
        self.initial_lr = initial_lr
        self.training_speeds = training_speeds
        self.testing_speeds = testing_speeds
        self.batch_size = batch_size
        self.report_name = report_name
        self.epochs = epochs
        self.PCA_type = PCA_selection_used
        self.reduced_testing_set = reduce_testing_set
        self.list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST",
                                "TA", "SO", "GM", "GL"]
        self.recorded_rmse_report = None
        self.recorded_acc_report = None
        self.list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
        self.evaluate_PCA_selection()

    def evaluate_PCA_selection(self):
        dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS03': 'R', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
        list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]
        PCA_selected_muscles = pd.read_csv('/media/ag6016/Storage/MuscleSelection/VarianceReports/'
                                           'PCA_' + self.PCA_type + '_var_selection.csv')
        report_idx = list(PCA_selected_muscles.index.values)
        recorded_idx = ['2 muscles', '3 muscles', '4 muscles', '5 muscles']
        recorded_PCA_rmse_values = pd.DataFrame(columns=self.list_of_subjects,
                                                index=['2 muscles', '3 muscles', '4 muscles', '5 muscles'])
        recorded_PCA_acc_values = pd.DataFrame(columns=self.list_of_subjects,
                                               index=['2 muscles', '3 muscles', '4 muscles', '5 muscles'])
        for subject in self.list_of_subjects:
            label_name = str(dominant_leg[subject]) + str(list_joint_angles[1])
            if subject == 'DS07':
                eliminate_speeds = ['07', '08']
                training_speeds = [speed for speed in self.training_speeds if speed not in eliminate_speeds]
                testing_speeds = [speed for speed in self.testing_speeds if speed not in eliminate_speeds]
            else:
                training_speeds = self.training_speeds
                testing_speeds = self.testing_speeds

            training_signals = []
            training_labels = []
            testing_signals = []
            testing_labels = []
            for i in range(len(training_speeds)):
                EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(subject, training_speeds[i],
                                                                          self.list_of_muscles, label_name=label_name)
                training_signals.append(EMG_signals)
                training_labels.append(labels)
            training_signals = np.concatenate(training_signals, axis=0)
            training_labels = np.concatenate(training_labels, axis=0)
            for i in range(len(testing_speeds)):
                EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(subject, testing_speeds[i],
                                                                          self.list_of_muscles,
                                                                          label_name=label_name)
                testing_signals.append(EMG_signals)
                testing_labels.append(labels)
            testing_signals = np.concatenate(testing_signals, axis=0)
            testing_labels = np.concatenate(testing_labels, axis=0)
            training_data = CNNLSTMDataPrep(training_signals, training_labels, window_length=512,
                                            window_step=40, batch_size=self.batch_size, sequence_length=15,
                                            label_delay=0,
                                            training_size=0.99, lstm_sequences=False, split_data=True,
                                            shuffle_full_dataset=True)
            x_train, y_train, _, _ = training_data.prepped_data  # shape (64, 15, 512, 400) and (64, 1, 400)

            average_values, std_values = training_data.norm_values
            testing_data = CNNLSTMDataPrep(testing_signals, testing_labels, window_length=512,
                                           window_step=40, batch_size=1, sequence_length=15, label_delay=0,
                                           training_size=0.99, lstm_sequences=False, split_data=True,
                                           shuffle_full_dataset=True)
            x_test, y_test, _, _ = testing_data.prepped_data
            for channel in range(x_train.shape[1]):
                x_train[:, channel, :, :] = torch.div((torch.sub(x_train[:, channel, :, :], average_values[channel])),
                                                   std_values[channel])
                x_test[:, channel, :, :] = torch.div((torch.sub(x_test[:, channel, :, :], average_values[channel])),
                                                  std_values[channel])
            if self.reduced_testing_set is not None:
                cut_off = int(x_test.shape[-1] / self.reduced_testing_set)
                x_test = x_test[:, :, :, 0:cut_off]
                y_test = y_test[:, :, 0:cut_off]
            for i in range(len(report_idx)):
                muscles_to_keep = PCA_selected_muscles.loc[report_idx[i], subject]
                muscles_to_keep_idx = [idx for idx, muscle in enumerate(self.list_of_muscles) if muscle in muscles_to_keep]
                updated_x_train = x_train.clone()
                updated_x_test = x_test.clone()
                for channel in range(x_train.shape[1]):
                    if channel in muscles_to_keep_idx:
                        pass
                    else:
                        updated_x_train[:, channel, :, :] = 0
                        updated_x_test[:, channel, :, :] = 0
                RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=self.epochs,
                                      saved_model_name='recent_PCA_evaluation', angle_range=90,
                                      initial_lr=self.initial_lr)
                RunningModel.train_network()
                recorded_PCA_rmse_values.loc[recorded_idx[i], subject] = RunningModel.recorded_validation_error
                recorded_PCA_acc_values.loc[recorded_idx[i], subject] = 1 - (RunningModel.recorded_validation_error /
                                                                       (torch.max(y_test).item() -
                                                                        torch.min(y_test).item()))
                self.recorded_rmse_report = recorded_PCA_rmse_values
                self.recorded_acc_report = recorded_PCA_acc_values
                self.recorded_rmse_report.to_csv(
                    '/media/ag6016/Storage/MuscleSelection/VarianceReports/PCA_RMSE_evaluation_' +
                    self.PCA_type + '_' + self.report_name + '.csv')
                self.recorded_acc_report.to_csv(
                    '/media/ag6016/Storage/MuscleSelection/VarianceReports/PCA_acc_evaluation_' +
                    self.PCA_type + '_' + self.report_name + '.csv')
                print(self.recorded_rmse_report)
                print(self.recorded_acc_report)

        self.recorded_rmse_report.to_csv('/media/ag6016/Storage/MuscleSelection/VarianceReports/PCA_RMSE_evaluation_' +
                                         self.PCA_type + '_' + self.report_name + '.csv')
        self.recorded_acc_report.to_csv('/media/ag6016/Storage/MuscleSelection/VarianceReports/PCA_acc_evaluation_' +
                                        self.PCA_type + '_' + self.report_name + '.csv')


class SelectionProcess:
    """
    The point of this algorithm is to train the network and then implement a one-out process to try and remove the
    electrodes one by one until there are only 4 left.
    To do this, we must:
    1) Record the overall accuracy of the network with all the electrodes
    2) Have a list of the electrodes names
    3) Remove all the electrodes one by one and retrain the network each time with the remaining electrodes
    4) Compare the accuracy and remove the electrode which has the least impact on accuracy (set the channel to 0)
    5) Repeat the process until only 4 electrodes are left
    6) Retain the order in which the electrodes were removed
    6) Repeat the process until all the subjects have been done
    """
    def __init__(self, subject, label_name, initial_lr, reserved_fraction=0.05, training_speeds=None, testing_speeds=None, saved_graph_name=None, batch_size=128, epochs=45, reduce_testing_set=None, model_type='TCN'):
        if subject == 'DS07':
            self.list_of_speeds = ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
        else:
            self.list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
        self.list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA",
                                "SO", "GM", "GL"]
        self.list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])
        self.subject = subject
        self.label_name = label_name
        self.reserved_fraction = reserved_fraction
        self.training_speeds = training_speeds
        self.testing_speeds = testing_speeds
        self.saved_name = saved_graph_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.reduced_test_set = reduce_testing_set
        self.model_type = model_type
        self.training_signals = None
        self.updated_training_signals = None
        self.training_labels = None
        self.testing_signals = None
        self.updated_testing_signals = None
        self.testing_labels = None
        self.reserved_signals = None
        self.reserved_labels = None
        self.performance_report = None
        self.muscles_used = [1] * len(self.list_of_muscles)
        self.general_report = pd.DataFrame(columns=list(self.list_of_muscles))
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.selection_report = pd.DataFrame(columns=self.list_of_muscles + ['Validation Loss', 'Validation accuracy'])

        if self.training_speeds is None:
            self.create_dataset_across_speeds()
        else:
            self.create_dataset_across_speeds_with_given_lists()
        # print("Data prep done")
        # self.train_model_with_all_channels()

    def get_speed_specific_profile_per_subject(self):
        columns = ['Speed', 'Training RMSE', 'Validation RMSE', 'Training accuracy', 'Validation accuracy', 'Epochs ran']
        speed_performance_profiler = pd.DataFrame(columns=columns)
        for speed in self.list_of_speeds:
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, speed, self.list_of_muscles,
                                                                      label_name=self.label_name)
            if self.model_type == 'TCN':
                prepped_data = CNNLSTMDataPrep(EMG_signals, labels, window_length=512, window_step=40,
                                               batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                               lstm_sequences=False, split_data=True, shuffle_full_dataset=False)
                x_train, y_train, x_test, y_test = prepped_data.prepped_data
                RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                      saved_model_name=self.subject + '_TCN_all_muscles_at_speed_' + speed,
                                      angle_range=90, load_model=False)
            elif self.model_type == 'MLP':
                prepped_data = CNNLSTMDataPrep(EMG_signals, labels, window_length=512, window_step=40,
                                               batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                               lstm_sequences=False, split_data=True, shuffle_full_dataset=False, filter_data=True)
                x_train, y_train, x_test, y_test = prepped_data.prepped_data
                RunningModel = RunMLP(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                      saved_model_name=self.subject + '_MLP_all_muscles_at_speed_' + speed,
                                      angle_range=90, load_model=False)
            else:
                raise Exception("This model type does not exist")
            RunningModel.train_network()
            new_row = pd.DataFrame([[speed, RunningModel.recorded_training_error, RunningModel.recorded_validation_error,
                                    RunningModel.recorded_training_error/(torch.max(y_train).item() - torch.min(y_train).item()),
                                    RunningModel.recorded_validation_error/(torch.max(y_test).item() - torch.min(y_test).item()),
                                    RunningModel.epochs_ran]], columns=columns)
            speed_performance_profiler = pd.concat([speed_performance_profiler, new_row], ignore_index=True)
        speed_performance_profiler.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject + '_' + self.model_type + '_all_speeds.csv')

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
            if self.model_type == 'TCN':
                prepped_data = CNNLSTMDataPrep(training_signals, training_labels, window_length=512, window_step=40,
                                               batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                               lstm_sequences=False, split_data=True, shuffle_full_dataset=False)
                x_train, y_train, x_test, y_test = prepped_data.prepped_data
                RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                      saved_model_name=self.subject + '_TCN_all_muscles_at_pace_' + paces[pace_idx],
                                      angle_range=90, load_model=False)
            elif self.model_type == 'MLP':
                prepped_data = CNNLSTMDataPrep(training_signals, training_labels, window_length=512, window_step=40,
                                               batch_size=32, sequence_length=15, label_delay=0, training_size=0.7,
                                               lstm_sequences=False, split_data=True, shuffle_full_dataset=False, filter_data=True)
                x_train, y_train, x_test, y_test = prepped_data.prepped_data
                RunningModel = RunMLP(x_train, y_train, x_test, y_test, n_channels=15, epochs=800,
                                      saved_model_name=self.subject + '_MLP_all_muscles_at_pace_' + paces[pace_idx],
                                      angle_range=90, load_model=False)
            else:
                raise Exception("This model type does not exist")
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

    def create_dataset_across_speeds_with_given_lists(self):
        training_signals = []
        training_labels = []
        testing_signals = []
        testing_labels = []
        for i in range(len(self.training_speeds)):
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, self.training_speeds[i],
                                                                      self.list_of_muscles, label_name=self.label_name)
            training_signals.append(EMG_signals)
            training_labels.append(labels)
        self.training_signals = np.concatenate(training_signals, axis=0)
        self.training_labels = np.concatenate(training_labels, axis=0)
        print("Training signals shape is ", self.training_signals.shape, self.training_labels.shape)
        for i in range(len(self.testing_speeds)):
            EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(self.subject, self.testing_speeds[i],
                                                                      self.list_of_muscles, label_name=self.label_name)
            testing_signals.append(EMG_signals)
            testing_labels.append(labels)
        self.testing_signals = np.concatenate(testing_signals, axis=0)
        self.testing_labels = np.concatenate(testing_labels, axis=0)
        print("Testing data shape ", self.testing_signals.shape, self.testing_labels.shape)

    def train_model_with_all_channels(self):
        if self.training_speeds is not None and self.testing_speeds is not None:
            if self.model_type == 'TCN':
                training_data = CNNLSTMDataPrep(self.training_signals, self.training_labels, window_length=512,
                                                window_step=40, batch_size=self.batch_size, sequence_length=15, label_delay=0,
                                                training_size=0.99, lstm_sequences=False, split_data=True,
                                                shuffle_full_dataset=True)
                testing_data = CNNLSTMDataPrep(self.testing_signals, self.testing_labels, window_length=512,
                                               window_step=40, batch_size=1, sequence_length=15, label_delay=0,
                                               training_size=0.99, lstm_sequences=False, split_data=True,
                                               shuffle_full_dataset=True)
            elif self.model_type == 'MLP':
                training_data = CNNLSTMDataPrep(self.training_signals, self.training_labels, window_length=512,
                                                window_step=40, batch_size=self.batch_size, sequence_length=15,
                                                label_delay=0,
                                                training_size=0.99, lstm_sequences=False, split_data=True,
                                                shuffle_full_dataset=True, filter_data=True)
                testing_data = CNNLSTMDataPrep(self.testing_signals, self.testing_labels, window_length=512,
                                               window_step=40, batch_size=1, sequence_length=15, label_delay=0,
                                               training_size=0.99, lstm_sequences=False, split_data=True,
                                               shuffle_full_dataset=True, filter_data=True)
            else:
                raise Exception("This model type does not exist")
            x_train, y_train, _, _ = training_data.prepped_data
            average_values, std_values = training_data.norm_values

            x_test, y_test, _, _ = testing_data.prepped_data
            for channel in range(x_train.shape[1]):
                x_train[:, channel, :] = torch.div((torch.sub(x_train[:, channel, :], average_values[channel])),
                                                   std_values[channel])
                x_test[:, channel, :] = torch.div((torch.sub(x_test[:, channel, :], average_values[channel])),
                                                  std_values[channel])
        else:
            if self.model_type == 'TCN':
                prepped_data = CNNLSTMDataPrep(self.training_signals, self.training_labels, window_length=512,
                                               window_step=40, batch_size=self.batch_size, sequence_length=15,
                                               label_delay=0, training_size=0.95, lstm_sequences=False, split_data=True,
                                               shuffle_full_dataset=True)
            elif self.model_type == 'MLP':
                prepped_data = CNNLSTMDataPrep(self.training_signals, self.training_labels, window_length=512,
                                               window_step=40,
                                               batch_size=self.batch_size, sequence_length=15, label_delay=0,
                                               training_size=0.95,
                                               lstm_sequences=False, split_data=True, shuffle_full_dataset=True,
                                               filter_data=True)
            else:
                raise Exception("This model type does not exist")
            x_train = prepped_data.x_train
            y_train = prepped_data.y_train
            y_test = prepped_data.y_test
            x_test = prepped_data.x_test
        if self.reduced_test_set is not None:
            cut_off = int(x_test.shape[-1]/self.reduced_test_set)
            x_test = x_test[:, :, :, 0:cut_off]
            y_test = y_test[:, :, 0:cut_off]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        if self.model_type == 'TCN':
            RunningModel = RunTCN(self.x_train, self.y_train, self.x_test, self.y_test, n_channels=15, epochs=self.epochs,
                                  saved_model_name='TCN_all_channels_' + self.subject + self.saved_name, angle_range=90,
                                  initial_lr=self.initial_lr)
        elif self.model_type == 'MLP':
            RunningModel = RunMLP(self.x_train, self.y_train, self.x_test, self.y_test, n_channels=15,
                                  epochs=self.epochs,
                                  saved_model_name='MLP_all_channels_' + self.subject + self.saved_name, angle_range=90,
                                  initial_lr=self.initial_lr)
        else:
            raise Exception("This model type does not exist")
        RunningModel.train_network()
        # self.performance_report = pd.DataFrame({'Number of channels': 15, 'Number of epochs': RunningModel.epochs_ran,
        #                                         'Training loss': RunningModel.recorded_training_error,
        #                                         'Validation loss': RunningModel.recorded_validation_error,
        #                                         'Training accuracy': 1 - (RunningModel.recorded_training_error /
        #                                                                   (torch.max(y_train).item() -
        #                                                                    torch.min(y_train).item())),
        #                                         'Validation accuracy': 1 - (RunningModel.recorded_validation_error /
        #                                                                     (torch.max(y_test).item() -
        #                                                                      torch.min(y_test).item())),
        #                                         'Electrode removed': 'None'}, index=[0])

        new_selection_row = self.muscles_used + [RunningModel.recorded_validation_error, 1 -
                                                 (RunningModel.recorded_validation_error / (torch.max(y_test).item() -
                                                                                            torch.min(y_test).item()))]
        new_row = pd.DataFrame([new_selection_row], columns=self.selection_report.columns, index=['15 electrodes'])
        self.selection_report = pd.concat([self.selection_report, new_row])


    def train_with_one_drop_out(self):
        self.train_model_with_all_channels()
        # self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject + '_' +
        #                                self.model_type + '_channel_selection_performance_' + self.saved_name + '.csv')
        if self.x_train is None:
            self.updated_training_signals = self.training_signals.copy()
            if self.testing_signals is not None:
                self.updated_testing_signals = self.testing_signals.copy()
            counter = self.training_signals.shape[-1]
            print(self.updated_training_signals.shape)
            print(self.updated_testing_signals.shape)
            while counter > 1:
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
                        train_signals[:, i] = 0
                        print("The training signals for this loop are ", train_signals[0, :])
                        if self.testing_speeds is not None:
                            test_signals = self.updated_testing_signals.copy()
                            test_signals[:, i] = 0
                            print("The testing signals for this loop are ", test_signals[0, :])
                            if self.model_type == 'TCN':
                                training_data = CNNLSTMDataPrep(self.training_signals, self.training_labels,
                                                                window_length=512,
                                                                window_step=40, batch_size=self.batch_size,
                                                                sequence_length=15, label_delay=0,
                                                                training_size=0.99, lstm_sequences=False,
                                                                split_data=True,
                                                                shuffle_full_dataset=True)
                                testing_data = CNNLSTMDataPrep(self.testing_signals, self.testing_labels,
                                                               window_length=512,
                                                               window_step=40, batch_size=1, sequence_length=15,
                                                               label_delay=0,
                                                               training_size=0.99, lstm_sequences=False,
                                                               split_data=True,
                                                               shuffle_full_dataset=True)
                            elif self.model_type == 'MLP':
                                training_data = CNNLSTMDataPrep(self.training_signals, self.training_labels,
                                                                window_length=512,
                                                                window_step=40, batch_size=self.batch_size,
                                                                sequence_length=15,
                                                                label_delay=0,
                                                                training_size=0.99, lstm_sequences=False,
                                                                split_data=True,
                                                                shuffle_full_dataset=True, filter_data=True)
                                testing_data = CNNLSTMDataPrep(self.testing_signals, self.testing_labels,
                                                               window_length=512,
                                                               window_step=40, batch_size=1, sequence_length=15,
                                                               label_delay=0,
                                                               training_size=0.99, lstm_sequences=False,
                                                               split_data=True,
                                                               shuffle_full_dataset=True, filter_data=True)
                            else:
                                raise Exception("This model type does not exist")
                            x_train, y_train, _, _ = training_data.prepped_data
                            average_values, std_values = training_data.norm_values
                            x_test, y_test, _, _ = testing_data.prepped_data
                            for channel in range(x_train.shape[1]):
                                x_train[:, channel, :] = torch.div(
                                    (torch.sub(x_train[:, channel, :], average_values[channel])),
                                    std_values[channel])
                                x_test[:, channel, :] = torch.div(
                                    (torch.sub(x_test[:, channel, :], average_values[channel])),
                                    std_values[channel])
                        else:
                            if self.model_type == 'TCN':
                                prepped_data = CNNLSTMDataPrep(self.training_signals, self.training_labels,
                                                               window_length=512,
                                                               window_step=40, batch_size=self.batch_size,
                                                               sequence_length=15,
                                                               label_delay=0, training_size=0.95, lstm_sequences=False,
                                                               split_data=True,
                                                               shuffle_full_dataset=True)
                            elif self.model_type == 'MLP':
                                prepped_data = CNNLSTMDataPrep(self.training_signals, self.training_labels,
                                                               window_length=512,
                                                               window_step=40,
                                                               batch_size=self.batch_size, sequence_length=15,
                                                               label_delay=0,
                                                               training_size=0.95,
                                                               lstm_sequences=False, split_data=True,
                                                               shuffle_full_dataset=True,
                                                               filter_data=True)
                            else:
                                raise Exception("This model type does not exist")
                            x_train, y_train, x_test, y_test = prepped_data.prepped_data
                        if self.model_type == 'TCN':
                            RunningModel = RunTCN(x_train, y_train, x_test, y_test, n_channels=15,
                                                  epochs=self.epochs,
                                                  saved_model_name='TCN_' + self.subject + '_' + str(counter-1) +
                                                                   '_electrodes_without_' + self.list_of_muscles[i],
                                                  angle_range=90,
                                                  initial_lr=self.initial_lr)
                        elif self.model_type == 'MLP':
                            RunningModel = RunMLP(x_train, y_train, x_test, y_test, n_channels=15,
                                                  epochs=self.epochs,
                                                  saved_model_name='MLP_' + self.subject + '_' + str(counter-1) +
                                                                   '_electrodes_without_' + self.list_of_muscles[i],
                                                  angle_range=90,
                                                  initial_lr=self.initial_lr)
                        else:
                            raise Exception("This model type does not exist")

                        RunningModel.train_network()
                        training_rmse.append(RunningModel.recorded_training_error)
                        validation_rmse.append(RunningModel.recorded_validation_error)
                        epochs_ran.append(RunningModel.epochs_ran)
                        print(training_rmse)
                        print(validation_rmse)
                        print(epochs_ran)
                        print("WE HAVE JUST FINISHED LOOP NUMBER ", i)

                # new_row = pd.DataFrame([training_rmse], columns=self.general_report.columns, index=['Training error with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])
                # new_row = pd.DataFrame([validation_rmse], columns=self.general_report.columns,
                #                        index=['Validation error with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])
                # new_row = pd.DataFrame([epochs_ran], columns=self.general_report.columns,
                #                        index=['Epochs ran with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])

                electrode_to_remove = validation_rmse.index(min(validation_rmse))
                print("The electrode to remove is ", electrode_to_remove)
                muscles_used = self.muscles_used
                muscles_used[electrode_to_remove] = 0
                self.muscles_used = muscles_used
                print(self.muscles_used)
                input("CHECK")
                # we want to remove the electrode whose absence has the least impact on model accuracy

                new_selection_row = self.muscles_used + [validation_rmse[electrode_to_remove], 1 -
                                                         (validation_rmse[electrode_to_remove] /
                                                          (np.max(self.training_labels) -
                                                           np.min(self.training_labels)))]
                new_row = pd.DataFrame([new_selection_row], columns=self.selection_report.columns,
                                       index=[counter-1])
                self.selection_report = pd.concat([self.selection_report, new_row])
                self.selection_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.model_type + '/' + self.subject +
                                             '_' + self.model_type + '_channel_selection_transition.csv')
                # new_row = pd.DataFrame([[counter-1, epochs_ran[electrode_to_remove], training_rmse[electrode_to_remove],
                #                         validation_rmse[electrode_to_remove], 1 - (training_rmse[electrode_to_remove] /
                #                                                                    (np.max(self.training_labels) -
                #                                                                     np.min(self.training_labels))),
                #                         1 - (validation_rmse[electrode_to_remove] / (np.max(self.training_labels) -
                #                                                                      np.min(self.training_labels))),
                #                         self.list_of_muscles[electrode_to_remove]]], columns=self.performance_report.columns)
                # self.performance_report = pd.concat([self.performance_report, new_row], ignore_index=True)

                self.updated_training_signals[:, electrode_to_remove] = 0
                self.updated_testing_signals[:, electrode_to_remove] = 0
                # self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                #                                '_channel_selection_performance_transition.csv')
                # self.general_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                #                            '_channel_info_transition.csv')
                counter -= 1

        else:
            self.updated_training_signals = self.x_train.clone()
            self.updated_testing_signals = self.x_test.clone()
            counter = self.training_signals.shape[1]
            print(self.updated_training_signals.shape)
            print(self.updated_testing_signals.shape)
            while counter > 1:
                training_rmse = []
                validation_rmse = []
                epochs_ran = []
                for i in range(len(self.list_of_muscles)):
                    # set the column to 0 to drop-out one of the electrodes
                    if torch.count_nonzero(self.updated_training_signals[:, i, :, :]) == 0:
                        training_rmse.append(999)
                        validation_rmse.append(999)
                        epochs_ran.append(999)
                    else:
                        x_train = self.updated_training_signals.clone()
                        x_train[:, i, :, :] = 0
                        x_test = self.updated_testing_signals.clone()
                        x_test[:, i, :, :] = 0
                        if self.model_type == 'TCN':
                            RunningModel = RunTCN(x_train, self.y_train, x_test, self.y_test, n_channels=15,
                                                  epochs=self.epochs,
                                                  saved_model_name='TCN_' + self.subject + '_' + str(counter-1) +
                                                                   '_electrodes_without_' + self.list_of_muscles[i],
                                                  angle_range=90,
                                                  initial_lr=self.initial_lr)
                        elif self.model_type == 'MLP':
                            RunningModel = RunMLP(x_train, self.y_train, x_test, self.y_test, n_channels=15,
                                                  epochs=self.epochs,
                                                  saved_model_name='MLP_' + self.subject + '_' + str(counter-1) +
                                                                   '_electrodes_without_' + self.list_of_muscles[i],
                                                  angle_range=90,
                                                  initial_lr=self.initial_lr)
                        else:
                            raise Exception("This model type does not exist")

                        RunningModel.train_network()
                        training_rmse.append(RunningModel.recorded_training_error)
                        validation_rmse.append(RunningModel.recorded_validation_error)
                        epochs_ran.append(RunningModel.epochs_ran)
                        print(training_rmse)
                        print(validation_rmse)
                        print(epochs_ran)
                        print("WE HAVE JUST FINISHED LOOP NUMBER ", i)

                # new_row = pd.DataFrame([training_rmse], columns=self.general_report.columns,
                #                        index=['Training error with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])
                # new_row = pd.DataFrame([validation_rmse], columns=self.general_report.columns,
                #                        index=['Validation error with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])
                # new_row = pd.DataFrame([epochs_ran], columns=self.general_report.columns,
                #                        index=['Epochs ran with ' + str(counter - 1) + ' electrodes'])
                # self.general_report = pd.concat([self.general_report, new_row])

                electrode_to_remove = validation_rmse.index(min(validation_rmse))
                print("The electrode to remove is ", electrode_to_remove)
                muscles_used = self.muscles_used
                muscles_used[electrode_to_remove] = 0
                self.muscles_used = muscles_used
                print(self.muscles_used)
                # we want to remove the electrode whose absence has the least impact on model accuracy

                new_selection_row = self.muscles_used + [validation_rmse[electrode_to_remove], 1 -
                                                         (validation_rmse[electrode_to_remove] /
                                                          (np.max(self.training_labels) -
                                                           np.min(self.training_labels)))]
                new_row = pd.DataFrame([new_selection_row], columns=self.selection_report.columns,
                                       index=[counter - 1])
                self.selection_report = pd.concat([self.selection_report, new_row])
                self.selection_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.model_type + '/' + self.subject +
                                             '_' + self.model_type + '_channel_selection_' + self.saved_name + 'csv')

                # new_row = pd.DataFrame(
                #     [[counter - 1, epochs_ran[electrode_to_remove], training_rmse[electrode_to_remove],
                #       validation_rmse[electrode_to_remove], 1 - (training_rmse[electrode_to_remove] /
                #                                                  (np.max(self.training_labels) -
                #                                                   np.min(self.training_labels))),
                #       1 - (validation_rmse[electrode_to_remove] / (np.max(self.training_labels) -
                #                                                    np.min(self.training_labels))),
                #       self.list_of_muscles[electrode_to_remove]]], columns=self.performance_report.columns)
                # self.performance_report = pd.concat([self.performance_report, new_row], ignore_index=True)

                self.updated_training_signals[:, electrode_to_remove, :, :] = 0
                self.updated_testing_signals[:, electrode_to_remove, :, :] = 0

                # self.performance_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                #                                '_channel_selection_performance_' + self.saved_name + '.csv')
                # self.general_report.to_csv('/media/ag6016/Storage/MuscleSelection/SubjectReports/' + self.subject +
                #                            '_channel_info_' + self.saved_name + '.csv')
                counter -= 1


if __name__ == "__main__":
    SelectionProcess('DS02', label_name='LKneeAngles')

