import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utility.dataprocessing import split_signals_into_TCN_windows, group_windows_into_sequences, shuffle, \
    split_into_train_test, split_into_batches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_the_predictions(model, saved_model_path, saved_model_name, x_test, y_test, lstm_hidden_size=16, lstm_layers=0):
    model.load_state_dict(torch.load(saved_model_path))
    predicted_y = []
    true_y = []
    with torch.no_grad():
        model.eval()
        if lstm_layers > 0:
            for rep in range(x_test.shape[-1]):
                h = torch.zeros(lstm_layers, x_test.shape[1], lstm_hidden_size).to(device)
                c = torch.zeros(lstm_layers, x_test.shape[1], lstm_hidden_size).to(device)
                hidden = tuple([h, c])
                for sequence in range(x_test.shape[0]):
                    predicted, hidden = model.forward(EMG_signal=(x_test[sequence, :, :, :, rep].float()),
                                                      hidden_tuple=tuple(hidden))
                    predicted_y.append(predicted.cpu().detach().numpy().squeeze())
                    true_y.append(y_test[sequence, :, :, rep].float().cpu().detach().numpy().squeeze())
        else:
            for rep in range(x_test.shape[-1]):
                predicted = model.forward(EMG_signal=(x_test[:, :, :, rep].float()))
                predicted_y.append(predicted.cpu().detach().numpy().squeeze())
                true_y.append(y_test[:, :, rep].float().cpu().detach().numpy().squeeze())
    plt.plot(predicted_y, label='Predicted angle')
    plt.plot(true_y, label='Ground truth')
    plt.legend()
    plt.title('Model prediction')
    plt.savefig('/media/ag6016/Storage/MuscleSelection/Images/' + saved_model_name + '.pdf', dpi=200)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

