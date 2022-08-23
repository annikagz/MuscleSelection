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

list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS03': 'R', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']#, 'R']
list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM", "GL"]
list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])
list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]
subject_idx = 0
speed_idx = 0
label_idx = 1


for i in range(1, len(list_of_subjects)):
    print("We are looking at subject ", list_of_subjects[i])
    label = str(dominant_leg[list_of_subjects[i]]) + str(list_joint_angles[1])
    selection_algorithm = SelectionProcess(list_of_subjects[i], label)
    selection_algorithm.train_with_one_drop_out()


