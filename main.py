import numpy as np
import c3d
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utility.dataprocessing import extract_c3d_data_to_hdf5, extract_hdf5_data_to_EMG_and_labels
from utility.plots import plot_the_predictions
from utility.conversions import normalise_signals, determine_synergies
from networks import CNNLSTMDataPrep, RunConvLSTM, RunTCN
from selectionprocess import SelectionProcess, PCAEvaluation

list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS03': 'R', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
list_of_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO", "GM", "GL"]
list_angles = list(["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"])
list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]
subject_idx = 0
speed_idx = 0
label_idx = 1


for i in range(len(list_of_subjects)):
    print("We are looking at subject ", list_of_subjects[i])
    label = str(dominant_leg[list_of_subjects[i]]) + str(list_joint_angles[1])
    selection_algorithm = SelectionProcess(list_of_subjects[i], label, initial_lr=0.000005,
                                           saved_graph_name='all_speeds', batch_size=64, epochs=100,
                                           reduce_testing_set=8, model_type='MLP')
    selection_algorithm.train_with_one_drop_out()

exit()


def get_all_signals_across_speeds(subject, list_of_speeds, label_name):
    list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO",
                       "GM", "GL"]
    training_signals = []
    training_labels = []
    for i in range(len(list_of_speeds) - 1):
        EMG_signals, labels = extract_hdf5_data_to_EMG_and_labels(subject, list_of_speeds[i], list_of_muscles,
                                                                  label_name=label_name)
        training_signals.append(EMG_signals)
        training_labels.append(labels)
    training_signals = np.concatenate(training_signals, axis=0)
    training_labels = np.concatenate(training_labels, axis=0)

    return training_signals, training_labels


# max_components_list = [5, 8, 11]
#
# for i in range(len(max_components_list)):
#     for subject_idx in range(len(list_of_subjects)):
#         label = str(dominant_leg[list_of_subjects[subject_idx]]) + str(list_joint_angles[1])
#         EMG_signals, labels = get_all_signals_across_speeds(list_of_subjects[subject_idx], list_of_speeds, label)
#         print(EMG_signals.shape, labels.shape)
#         range_components = max_components_list[i]
#         n_components = determine_synergies(EMG_signals, range_components)
#         plt.suptitle(list_of_subjects[subject_idx])
#         plt.savefig('/media/ag6016/Storage/MuscleSelection/synergies/'+list_of_subjects[subject_idx] + '_' +
#                     str(n_components) + '_synergies.pdf')
#         plt.show(block=False)
#         plt.pause(3)
#         plt.close()
#
# exit()

# print("We are training the fast vs medium speed")
# training_speeds = ['15', '16', '17', '18']
# for i in range(len(list_of_subjects)):
#     if i == 5:
#         testing_speeds = ['09', '10']
#         split_testing_set = 4
#     else:
#         testing_speeds = ['07', '08', '09', '10']
#         split_testing_set = 8
#     print("We are looking at subject ", list_of_subjects[i])
#     label = str(dominant_leg[list_of_subjects[i]]) + str(list_joint_angles[1])
#     selection_algorithm = SelectionProcess(list_of_subjects[i], label, initial_lr=0.000005,
#                                            training_speeds=training_speeds, testing_speeds=testing_speeds,
#                                            saved_graph_name='tested_on_slow', batch_size=64, epochs=30,
#                                            reduce_testing_set=split_testing_set)
#     selection_algorithm.train_with_one_drop_out()
#
#
# print("We are training the fast vs medium speed")
# training_speeds = ['15', '16', '17', '18']
# testing_speeds = ['11', '12', '13', '14']
# for i in range(len(list_of_subjects)):
#     print("We are looking at subject ", list_of_subjects[i])
#     label = str(dominant_leg[list_of_subjects[i]]) + str(list_joint_angles[1])
#     selection_algorithm = SelectionProcess(list_of_subjects[i], label, initial_lr=0.000005,
#                                            training_speeds=training_speeds, testing_speeds=testing_speeds,
#                                            saved_graph_name='tested_on_medium', batch_size=64, epochs=30,
#                                            reduce_testing_set=8)
#     selection_algorithm.train_with_one_drop_out()



# ======================================================================================================================
# list_of_subjects = ['DS01', 'DS07']
# training_speeds = ['15', '16', '17', '18']
#
# print("WE ARE TRAINING THE MUSCLE SELECTION TRAINED AND TESTED ON ALL STEADY-STATE SPEEDS")
# for subject_idx in range(len(list_of_subjects)):
#     print("Now training for subject ", list_of_subjects[subject_idx])
#     label = str(dominant_leg[list_of_subjects[subject_idx]]) + str(list_joint_angles[1])
#     selection_algorithm = SelectionProcess(list_of_subjects[subject_idx], label, initial_lr=0.001, saved_graph_name='original',
#                                            batch_size=128, epochs=40, reduce_testing_set=10)
#     selection_algorithm.train_with_one_drop_out()
#
#
# for subject_idx in range(len(list_of_subjects)):
#     label = str(dominant_leg[list_of_subjects[subject_idx]]) + str(list_joint_angles[1])
#     selection_algorithm = SelectionProcess(list_of_subjects[subject_idx], label, initial_lr=0.001, batch_size=32, epochs=800)
#     selection_algorithm.get_speed_specific_profile_per_subject()


# list_of_subjects = ['DS05', 'DS07']
# print("WE ARE TRAINING THE MUSCLE SELECTION TRANSITION")
# for subject_idx in range(len(list_of_subjects)):
#     if subject_idx == 0:
#         lr = 0.000001
#         speeds = list_of_speeds[0:-1]
#     else:
#         lr = 0.00001
#         speeds = list_of_speeds[2:-1]
#     print("Now training for subject ", list_of_subjects[subject_idx])
#     label = str(dominant_leg[list_of_subjects[subject_idx]]) + str(list_joint_angles[1])
#     selection_algorithm = SelectionProcess(list_of_subjects[subject_idx], label, initial_lr=lr, training_speeds=speeds,
#                                            testing_speeds=['R'], saved_graph_name='transition', batch_size=128, epochs=50)
#     selection_algorithm.train_with_one_drop_out()

# list_of_subjects = ['DS01']
# speeds = list_of_speeds[0:-1]
# for subject_idx in range(len(list_of_subjects)):
#     label = str(dominant_leg[list_of_subjects[subject_idx]]) + str(list_joint_angles[1])
#     # selection_algorithm = SelectionProcess(list_of_subjects[subject_idx], label, initial_lr=0.001, batch_size=32, epochs=800)
#     # selection_algorithm.get_speed_specific_profile_per_subject()
#     selection_algorithm = SelectionProcess(list_of_subjects[subject_idx], label, initial_lr=0.00001, training_speeds=speeds,
#                                            testing_speeds=['R'], saved_graph_name='transition', batch_size=128, epochs=50)
#     selection_algorithm.train_with_one_drop_out()


training_speeds = ['15', '16', '17', '18']
testing_speeds = ['07', '08', '09', '10']
PCAEvaluation(initial_lr=0.000005, training_speeds=training_speeds, testing_speeds=testing_speeds,
              report_name='tested_on_slow', batch_size=64, epochs=30, PCA_selection_used='min', reduce_testing_set=4)

testing_speeds = ['11', '12', '13', '14']
PCAEvaluation(initial_lr=0.000005, training_speeds=training_speeds, testing_speeds=testing_speeds,
              report_name='tested_on_med', batch_size=64, epochs=30, PCA_selection_used='min', reduce_testing_set=4)

training_speeds = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']
testing_speeds = ['R']
PCAEvaluation(initial_lr=0.00001, training_speeds=training_speeds, testing_speeds=testing_speeds,
              report_name='tested_on_transient', batch_size=128, epochs=50, PCA_selection_used='min')
