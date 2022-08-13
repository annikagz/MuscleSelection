import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utility.dataprocessing import split_signals_into_TCN_windows, group_windows_into_sequences, shuffle, \
    split_into_train_test, split_into_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNLSTMDataPrep:
    """
    For the data processing, with an input data of shape (n_samples, n_channels), we want to:
    1) Split the time series data along its length into the windows with steps: (window_length, n_channels, n_windows)
    2) Group these into the sequences along which the LSTM will remember things
    (sequence_length, window_length, n_channels, n_sequences (=n_windows/sequence_length)
    3) Shuffle this along the last axis, so the n_sequences axis
    4) Split into test and train (0.8)
    5) Batch up the training data, so that the x_train is of shape:
    (batch_size, sequence_length, window_length, n_channels, n_batches), and the testing data is of shape:
    (1, sequence_length, 1, 1, n_sequences)
    """
    def __init__(self, EMG_signals, labels, window_length, window_step, batch_size, sequence_length=10, label_delay=0,
                 training_size=0.9, split_data=True, shuffle_full_dataset=True):
        self.EMG_signals = EMG_signals
        self.labels = labels
        self.n_channels = self.EMG_signals.shape[-1]
        self.window_length = window_length
        self.window_step = window_step
        self.prediction_delay = label_delay
        self.batch_size = batch_size
        self.sequence_length = sequence_length  # corresponds to the number of windows per sequence
        self.training_size = training_size

        # NORMALISE THE SIGNAL -----------------------------------------------------------------------------------------
        for channel in range(self.EMG_signals.shape[-1]):
            self.EMG_signals[:, channel] = self.EMG_signals[:, channel] \
                                           / (0.95 * np.max(np.abs(self.EMG_signals[:, channel])))

        # WINDOW THE SIGNAL --------------------------------------------------------------------------------------------
        self.windowed_signals, self.windowed_labels = split_signals_into_TCN_windows\
            (self.EMG_signals, self.labels, self.window_length, self.window_step, self.prediction_delay, False)
        # windowed signals of shape (window_length, n_channels, n_reps)
        # windowed labels of shape (n_channels, n_reps)

        # GROUP INTO SEQUENCES -----------------------------------------------------------------------------------------
        self.windowed_signals, self.windowed_labels = group_windows_into_sequences\
            (self.windowed_signals, self.windowed_labels, self.sequence_length, window_axis=-1)
        # windowed signals of shape (n_windows_per_sequence, window_length, n_channels, n_sequences)
        # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

        # TRANSPOSE ----------------------------------------------------------------------------------------------------
        self.windowed_signals = self.windowed_signals.transpose((0, 2, 1, 3))
        # windowed signals of shape (n_windows_per_sequence, n_channels, window_length, n_sequences)
        # windowed labels of shape (n_windows_per_sequence, n_channels, n_sequences)

        if shuffle_full_dataset:
            self.windowed_signals, self.windowed_labels = shuffle(self.windowed_signals, self.windowed_labels)

        if split_data:
            # SPLIT INTO TRAIN-TEST ------------------------------------------------------------------------------------
            self.x_train, self.x_test, self.y_train, self.y_test = split_into_train_test\
                (self.windowed_signals, self.windowed_labels, train_size=self.training_size, split_axis=-1)
            # x train of shape (sequence_length, n_channels, window_length, n_training_sequences)
            # y train of shape (sequence_length, n_channels, n_training_sequences)
            # x test of shape (sequence_length, n_channels, window_length, n_testing_sequences)
            # y test of shape (sequence_length, n_channels, n_testing_sequences)

            # SHUFFLE ALONG THE N_SEQUENCE AXIS ------------------------------------------------------------------------
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)

            # BATCH THE TRAINING DATA ----------------------------------------------------------------------------------
            self.x_train, self.y_train, self.x_test, self.y_test = split_into_batches(self.x_train, self.y_train,
                                                                                      self.x_test, self.y_test,
                                                                                      self.batch_size, batch_axis=1)
            # x train of shape (sequence_length, batch_size, n_channels, window_length, n_batches)
            # y train of shape (sequence_length, batch_size, n_channels, n_batches)
            # x test of shape (sequence_length, 1, n_channels, window_length, n_testing_sequences)
            # y test of shape (sequence_length, 1, 1, n_testing_sequences)

        # TURN INTO TENSORS --------------------------------------------------------------------------------------------
        self.turn_into_tensors()

        # PRODUCE A FINAL ATTRIBUTE WITH ALL THE RELEVANT INFORMATION TO BE EASILY EXTRACTED ---------------------------
        self.prepped_data = self.x_train, self.x_test, self.y_train, self.y_test

    def turn_into_tensors(self):
        self.x_train = torch.autograd.Variable(torch.from_numpy(self.x_train), requires_grad=True).to(device)
        self.y_train = torch.autograd.Variable(torch.from_numpy(self.y_train), requires_grad=True).to(device)
        self.x_test = torch.from_numpy(self.x_test).to(device)
        self.y_test = torch.from_numpy(self.y_test).to(device)


class ConvLSTMNetwork(nn.Module):
    """
    The input to this network is going to be of shape:
    x_train : (sequence_length, batch_size, n_channels, window_length, n_reps)
    y_train: (sequence_length, batch_size, 1, n_reps)
    x_test : (sequence_length, 1, n_channels, window_length, n_reps)
    y_test : (sequence_length, 1, 1, n_reps)
    input shape for nn.Conv1D : (batch_size, n_channels, window_length)
    input shape for nn.LSTM: (batch_size, window_length, input_size) if batch_first = True
    h, c shape for LSTM: (1 * n_LSTM_layers, input_size)

    The pseudo-code to implement is:

    for rep in x_train.shape[-1]:
        initialise h,c of the LSTM
        for sequence in x_train.shape[0]:
            out = CNN(x_train[sequence, :, :, :, rep)
            out, (h, c) = LSTM(out)
            predicted_output = flatten(out)
            loss = criterion(predicted_output, y_train[sequence, :, :, rep)
    """
    def __init__(self, n_inputs=15, kernel_size=5, dilation=4, stride=1, dropout=0.2, lstm_hidden_size=16,
                 lstm_n_layers=1):
        super(ConvLSTMNetwork, self).__init__()
        self.input_size = n_inputs
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_n_layers
        self.flattened_length = 1008
        self.dropout = dropout

        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=16, kernel_size=kernel_size, stride=stride,
                      dilation=dilation, padding='same'),  # (1, 8, 512)
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (1, 8, 256)
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (1, 16, 128)
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)  # (1, 16, 64)
        )

        self.LSTM = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.lstm_layers,
                            batch_first=True)

        self.DenseLayers = nn.Sequential(
            nn.Linear(self.flattened_length, int(self.flattened_length/2)),  # (1, 512)
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.flattened_length / 2), int(self.flattened_length / 4)),  # (1, 256)
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.flattened_length / 4), int(self.flattened_length / 8)),  # (1, 128)
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.flattened_length / 8), 1),
            nn.Sigmoid()
        )

    def forward(self, EMG_signal, hidden_tuple):
        # (batch, n_channels, window_length)
        hidden_tuple = tuple([each.data for each in hidden_tuple])
        out = self.CNN(EMG_signal)  # output shape is (16, 16, 64)
        self.hidden_size = int(out.shape[1])
        out = torch.transpose(out, -2, -1)
        out, hidden_tuple = self.LSTM(out, hidden_tuple)
        out = nn.ReLU()(out)
        out = torch.transpose(out, -2, -1)
        out = nn.Flatten()(out)
        self.flattened_length = out.shape[-1]
        out = self.DenseLayers(out)*70
        return out, hidden_tuple


class RunConvLSTM:
    def __init__(self, x_train, y_train, x_test, y_test, n_channels, lstm_hidden, epochs, saved_model_name, lstm_layers):
        self.hidden_size = lstm_hidden
        self.lstm_layers = lstm_layers
        self.model = ConvLSTMNetwork(n_inputs=n_channels, kernel_size=5, stride=1, dilation=3, dropout=0.1,
                                     lstm_hidden_size=self.hidden_size, lstm_n_layers=lstm_layers).to(device)
        self.model_type = 'CNN-LSTM'
        self.saved_model_name = saved_model_name
        self.saved_model_path = '/media/ag6016/Storage/MuscleSelection/Models/' + self.saved_model_name + '.pth'
        self.criterion = nn.MSELoss().to(device)
        self.epochs = epochs
        self.writer = SummaryWriter()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.recorded_training_error = None
        self.recorded_validation_error = None
        self.epochs_ran = 0

    def train_network(self):
        rep_step = 0
        lowest_error = 1000.0
        cut_off_counter = 0
        for epoch in range(self.epochs):
            print("Epoch number:", epoch)
            if epoch < 10:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
            elif 10 < epoch < 20:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005, betas=(0.9, 0.999))
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999))
            running_training_loss = 0.0
            running_validation_loss = 0.0
            for rep in tqdm(np.arange(self.x_train.shape[-1])):
                h = torch.zeros(self.lstm_layers, self.x_train.shape[1], self.hidden_size).to(device)
                c = torch.zeros(self.lstm_layers, self.x_train.shape[1], self.hidden_size).to(device)
                hidden = tuple([h, c])
                for sequence in range(self.x_train.shape[0]):
                    predicted, hidden = self.model.forward(EMG_signal=(self.x_train[sequence, :, :, :, rep].float()),
                                                           hidden_tuple=tuple(hidden))
                    loss = self.criterion(predicted, self.y_train[sequence, :, :, rep].float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.writer.add_scalar("Training loss " + self.saved_model_name, loss, global_step=rep_step)
                    running_training_loss += loss.item()
                    rep_step += 1
            recorded_training_error = running_training_loss / (self.x_train.shape[0] * self.x_train.shape[-1])

            # VALIDATION LOOP
            with torch.no_grad():
                for rep in range(self.x_test.shape[-1]):
                    h = torch.zeros(1, self.x_test.shape[1], self.hidden_size).to(device)
                    c = torch.zeros(1, self.x_test.shape[1], self.hidden_size).to(device)
                    hidden = tuple([h, c])
                    for sequence in range(self.x_test.shape[0]):
                        predicted, hidden = self.model.forward(
                            EMG_signal=(self.x_test[sequence, :, :, :, rep].float()),
                            hidden_tuple=tuple(hidden))
                        validation_loss = self.criterion(predicted, self.y_test[sequence, :, :, rep].float())
                        running_validation_loss += validation_loss.item()

            recorded_validation_error = running_validation_loss / (self.x_test.shape[0] * self.x_test.shape[-1])
            if recorded_validation_error < lowest_error:
                torch.save(self.model.state_dict(), self.saved_model_path)
                lowest_error = recorded_validation_error
                self.recorded_validation_error = recorded_validation_error
                self.recorded_training_error = recorded_training_error
                self.epochs_ran = epoch
            else:
                cut_off_counter += 1

            # STOP THE MODEL WHEN IT IS NO LONGER LEARNING
            if cut_off_counter > 3:
                break


class CustomLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers=1):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.fc1 = nn.Linear(self.n_features, self.hidden_size)
        self.conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=2,
                              padding='same', dilation=2)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size, self.n_features)
        self.fc3 = nn.Linear(self.n_features, self.n_features)
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x, hidden_tuple):
        hidden_tuple = tuple([each.data for each in hidden_tuple])
        # input shape (batch_size, sequence_length, n_channels)
        #out = self.act1(self.fc1(x.float()))
        # shape (batch_size, sequence_length, n_channels)
        #out = torch.transpose(out, -2, -1)
        # shape (batch_size, n_channels, sequence_length)
        out = self.conv(x.float())
        # shape (batch_size, n_channels, sequence_length)
        out = torch.transpose(out, -2, -1)
        # shape (batch_size, sequence_length, n_channels)
        out = self.act1(out)
        # shape (batch_size, sequence_length, n_channels)
        out, hidden = self.lstm(out, hidden_tuple)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # shape (batch_size, sequence_length, n_channels)
        out = self.act2(self.fc2(out))
        out = torch.transpose(out, -2, -1)
        # shape (batch_size, n_channels, sequence_length)
        out = self.flatten
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, hidden