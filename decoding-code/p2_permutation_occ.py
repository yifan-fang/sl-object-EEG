import numpy as np
import os
import pandas as pd
import scipy.io as sio
import re
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

print("Loading EEG and Behavioral data...")
os.chdir('filteredEEGInfrequent')
all_sub_EEG_inf = {}
for file in os.listdir():
    mat_data = sio.loadmat(file)
    loaded_matrix = mat_data['eegKept']
    subID = re.findall(r'\d+', file)[0]
    all_sub_EEG_inf[subID] = loaded_matrix

os.chdir('../BehavioralDataAlignedInfrequent')
all_sub_behav_inf = {}
for file in os.listdir():
    subID = re.findall(r'\d+', file)[0]
    all_sub_behav_inf[subID] = pd.read_csv(file)

os.chdir('../filteredEEGTrainInfrequent')
all_sub_EEG_train_inf = {}
for file in os.listdir():
    mat_data = sio.loadmat(file)
    loaded_matrix = mat_data['eegKept']
    subID = re.findall(r'\d+', file)[0]
    all_sub_EEG_train_inf[subID] = loaded_matrix

os.chdir('../BehavioralDataAlignedTrainInfrequent')
all_sub_behav_train_inf = {}
for file in os.listdir():
    subID = re.findall(r'\d+', file)[0]
    all_sub_behav_train_inf[subID] = pd.read_csv(file)

all_sub_EEG_phase2_inf = {}
for sub, matrix in all_sub_EEG_train_inf.items():
    trainEnd = matrix.shape[2]
    EEG_test_inf = all_sub_EEG_inf[sub][...,trainEnd:]
    all_sub_EEG_phase2_inf[sub] = EEG_test_inf
    
all_sub_behav_phase2_inf = {}
for sub, df in all_sub_behav_train_inf.items():
    trainEnd = len(df)
    behav_test_inf = all_sub_behav_inf[sub].iloc[trainEnd:].reset_index(drop=True)
    all_sub_behav_phase2_inf[sub] = behav_test_inf

os.chdir('../filteredEEGFrequent')
all_sub_EEG_f = {}
for file in os.listdir():
    mat_data = sio.loadmat(file)
    loaded_matrix = mat_data['eegKept']
    subID = re.findall(r'\d+', file)[0]
    all_sub_EEG_f[subID] = loaded_matrix

os.chdir('../BehavioralDataAlignedFrequent')
all_sub_behav_f = {}
for file in os.listdir():
    subID = re.findall(r'\d+', file)[0]
    all_sub_behav_f[subID] = pd.read_csv(file)

os.chdir('../filteredEEGTrainFrequent')
all_sub_EEG_train_f = {}
for file in os.listdir():
    mat_data = sio.loadmat(file)
    loaded_matrix = mat_data['eegKept']
    subID = re.findall(r'\d+', file)[0]
    all_sub_EEG_train_f[subID] = loaded_matrix

os.chdir('../BehavioralDataAlignedTrainFrequent')
all_sub_behav_train_f = {}
for file in os.listdir():
    subID = re.findall(r'\d+', file)[0]
    all_sub_behav_train_f[subID] = pd.read_csv(file)

os.chdir('../')
all_sub_EEG_phase2_f = {}
for sub, matrix in all_sub_EEG_train_f.items():
    trainEnd = matrix.shape[2]
    EEG_test_f = all_sub_EEG_f[sub][...,trainEnd:]
    all_sub_EEG_phase2_f[sub] = EEG_test_f

all_sub_behav_phase2_f = {}
for sub, df in all_sub_behav_train_f.items():
    trainEnd = len(df)
    behav_test_f = all_sub_behav_f[sub].iloc[trainEnd:].reset_index(drop=True)
    all_sub_behav_phase2_f[sub] = behav_test_f
print("Data loading complete.")


def run_one_timepoint(t, subEEG, train_idx, test_idx, y_shuffled_combined, mask_array_f, mask_array_inf):
    X = subEEG[:, t, :].T
    X_train = X[train_idx]
    X_test = X[test_idx]

    # replace missing value with mean
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_test_f = X_test[mask_array_f]
    X_test_inf = X_test[mask_array_inf]
    # using the same shuffled labels for both conditions to ensure the permutation is consistent
    y_train = y_shuffled_combined[train_idx]
    y_test = y_shuffled_combined[test_idx]
    y_test_f = y_test[mask_array_f]
    y_test_inf = y_test[mask_array_inf]

    # Train SVM
    clf = SVC(decision_function_shape='ovr', kernel='linear')
    clf.fit(X_train, y_train)

    # Predict
    # y_pred = clf.predict(X_test)
    y_pred_f = clf.predict(X_test_f)
    y_pred_inf = clf.predict(X_test_inf)

    # Model accuracy
    # overall_accuracy = accuracy_score(y_test, y_pred)
    acc_array_f = accuracy_score(y_test_f, y_pred_f)
    acc_array_inf = accuracy_score(y_test_inf, y_pred_inf)

    # return overall_accuracy, acc_array_f, acc_array_inf
    return acc_array_f, acc_array_inf





def process_single_subject(single_subject, eeg_f_data, eeg_inf_data, behav_f_data, behav_inf_data):
    print(f"Starting processing for subject: {single_subject}")

    EEG_f = eeg_f_data[single_subject]
    EEG_inf = eeg_inf_data[single_subject]
    behav_f = behav_f_data[single_subject]
    behav_inf = behav_inf_data[single_subject]

    min_trialN = np.min([len(behav_f), len(behav_inf)])

    permN = 1000
    perm = []
    perm_f = []
    perm_inf = [] 
    # only using occipital electrodes, dropping: diode, HEOG, M1, FP1, FP2, F3, Fz, F4, FC5, FC1, FC2, FC6, T7, C3, C4, T8
    # python index starts with 0, all -1 from MATLAB label
    drop_electrodes = [4, 31, 32, 0, 30, 1, 28, 29, 30, 3, 2, 27, 27, 5, 6, 24, 25] 

    for i in range(permN):
        # Sample same number of trials for both conditions
        idx_f = np.random.choice(len(behav_f), min_trialN, replace=False)
        idx_inf = np.random.choice(len(behav_inf), min_trialN, replace=False)
        EEG_f_sampled = EEG_f[:, :, idx_f]
        EEG_inf_sampled = EEG_inf[:, :, idx_inf]
        behav_f_sampled = behav_f.iloc[idx_f]
        behav_inf_sampled = behav_inf.iloc[idx_inf]

        subEEG_combined = np.concatenate([EEG_f_sampled, EEG_inf_sampled], axis=2)

        subEEG = np.delete(subEEG_combined, drop_electrodes, axis=1)
        # select relevant timepoints (-200-600ms)
        x_ms = np.linspace(-300, 800, subEEG.shape[1])
        t_start = -200
        t_end   = 600
        mask = (x_ms >= t_start) & (x_ms <= t_end)
        time_ms   = x_ms[mask] 
        timepoint_indices = np.where(mask)[0]

        n_trials_f = EEG_f_sampled.shape[2]
        n_trials_inf = EEG_inf_sampled.shape[2]
        origin_labels = np.concatenate([
            np.zeros(n_trials_f),  # 0 for f
            np.ones(n_trials_inf)    # 1 for inf
        ])

        y_f = behav_f_sampled['TargetLocation']  
        y_inf = behav_inf_sampled['TargetLocation'] 
        y_combined = np.concatenate([y_f, y_inf])

        # permutation
        y_shuffled_combined = np.random.permutation(y_combined)

        n_total_trials = n_trials_f + n_trials_inf
        indices = np.arange(n_total_trials)

        train_idx, test_idx = train_test_split(
            indices, 
            test_size=0.3, 
            random_state=42
        )

        origin_test = origin_labels[test_idx]
        mask_array_f = origin_test == 0
        mask_array_inf = origin_test == 1


        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_one_timepoint)(
                t, subEEG, train_idx, test_idx, y_shuffled_combined, mask_array_f, mask_array_inf
            ) for t in timepoint_indices
        )

        # acc_overall, acc_f, acc_inf = map(np.array, zip(*results))
        acc_f, acc_inf = map(np.array, zip(*results))
        # perm.append(acc_overall)
        perm_f.append(acc_f)
        perm_inf.append(acc_inf)
        

    # np.save(f"phase2_occ_permutations/{single_subject}.npy", perm)
    np.save(f"phase2_occ_permutations_f/{single_subject}.npy", perm_f)
    np.save(f"phase2_occ_permutations_inf/{single_subject}.npy", perm_inf)

    print(f"Finished and saved results for subject: {single_subject}")

if __name__ == "__main__":
    subjects = ['39', '40', '41', '42', '44', '45',
                '46', '47', '48', '49']
    
    # Create output directories if they don't exist
    # os.makedirs("phase2_occ_permutations", exist_ok=True)
    os.makedirs("phase2_occ_permutations_f", exist_ok=True)
    os.makedirs("phase2_occ_permutations_inf", exist_ok=True)

    for single_subject in subjects:
        process_single_subject(single_subject, all_sub_EEG_phase2_f, all_sub_EEG_phase2_inf,
            all_sub_behav_phase2_f, all_sub_behav_phase2_inf)

    print("All subjects have been processed.")
