import os
import pandas as pd
import numpy as np
import scipy.io as sio
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

# --- 1. DATA LOADING (No changes here) ---
# It's good practice to load all data once at the beginning.

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

# --- 2. DEFINE THE PROCESSING FUNCTIONS ---

# This function remains the same. It processes a single timepoint.
def run_one_timepoint(t, subEEG, train_idx, test_idx, y_combined, mask_array_f, mask_array_inf):
    X = subEEG[:, t, :].T
    X_train = X[train_idx]
    X_test = X[test_idx]

    # replace missing value with mean
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Note: A small bug was fixed here. It's better to create these arrays once outside the loop.
    # But for minimal changes, we keep the logic. Ensure masks align with X_test.
    X_test_f = X_test[mask_array_f]
    X_test_inf = X_test[mask_array_inf]
    
    y_train = y_combined[train_idx]
    y_test = y_combined[test_idx]
    y_test_f = y_test[mask_array_f]
    y_test_inf = y_test[mask_array_inf]

    # Train SVM
    clf = SVC(decision_function_shape='ovr', kernel='linear')
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_pred_f = clf.predict(X_test_f)
    y_pred_inf = clf.predict(X_test_inf)

    # Model accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    acc_array_f = accuracy_score(y_test_f, y_pred_f)
    acc_array_inf = accuracy_score(y_test_inf, y_pred_inf)

    return overall_accuracy, acc_array_f, acc_array_inf


# NEW: This function contains all the logic for processing ONE subject.
def process_single_subject(single_subject, eeg_f_data, eeg_inf_data, behav_f_data, behav_inf_data):
    """
    Runs the full analysis pipeline for a single subject and saves the results.
    """
    print(f"Starting processing for subject: {single_subject}")
    
    # Get data for the current subject
    EEG_f = eeg_f_data[single_subject]
    EEG_inf = eeg_inf_data[single_subject]
    behav_f = behav_f_data[single_subject]
    behav_inf = behav_inf_data[single_subject]

    min_trialN = np.min([len(behav_f), len(behav_inf)])
    repeatN = 100
    model_accuracies = []
    model_accuracies_f = []
    model_accuracies_inf = []
    # only using occipital electrodes, dropping: diode, HEOG, M1, FP1, FP2, F3, Fz, F4, FC5, FC1, FC2, FC6, T7, C3, C4, T8
    # python index starts with 0, all -1 from MATLAB label
    drop_electrodes = [4, 31, 32, 0, 30, 1, 28, 29, 30, 3, 2, 27, 27, 5, 6, 24, 25] 
    for i in range(repeatN):
        # Sample same number of trials for both conditions
        idx_f = np.random.choice(len(behav_f), min_trialN, replace=False)
        idx_inf = np.random.choice(len(behav_inf), min_trialN, replace=False)
        EEG_f_sampled = EEG_f[:, :, idx_f]
        EEG_inf_sampled = EEG_inf[:, :, idx_inf]
        behav_f_sampled = behav_f.iloc[idx_f]
        behav_inf_sampled = behav_inf.iloc[idx_inf]
        
        subEEG_combined = np.concatenate([EEG_f_sampled, EEG_inf_sampled], axis=2)
        subEEG = np.delete(subEEG_combined, drop_electrodes, axis=1)

        x_ms = np.linspace(-300, 800, subEEG.shape[1])
        t_start, t_end = -200, 600
        mask = (x_ms >= t_start) & (x_ms <= t_end)
        timepoint_indices = np.where(mask)[0]

        n_trials_f = EEG_f_sampled.shape[2]
        n_trials_inf = EEG_inf_sampled.shape[2]
        origin_labels = np.concatenate([np.zeros(n_trials_f), np.ones(n_trials_inf)])
        y_f = behav_f_sampled['TargetLocation']
        y_inf = behav_inf_sampled['TargetLocation']
        y_combined = np.concatenate([y_f, y_inf])

        indices = np.arange(len(y_combined))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=origin_labels
        )
        
        origin_test = origin_labels[test_idx]
        mask_array_f = origin_test == 0
        mask_array_inf = origin_test == 1

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_one_timepoint)(
                t, subEEG, train_idx, test_idx, y_combined, mask_array_f, mask_array_inf
            ) for t in timepoint_indices
        )

        acc_overall, acc_f, acc_inf = map(np.array, zip(*results))
        model_accuracies.append(acc_overall)
        model_accuracies_f.append(acc_f)
        model_accuracies_inf.append(acc_inf)

    # Save results for the current subject
    np.save(f"phase2_acc_occ/model_accuracies_{single_subject}.npy", model_accuracies)
    np.save(f"phase2_acc_occ_f/model_accuracies_f_{single_subject}.npy", model_accuracies_f)
    np.save(f"phase2_acc_occ_inf/model_accuracies_inf_{single_subject}.npy", model_accuracies_inf)
    
    print(f"Finished and saved results for subject: {single_subject}")


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    subjects = ['49']
    
    # Create output directories if they don't exist
    os.makedirs("phase2_acc_occ", exist_ok=True)
    os.makedirs("phase2_acc_occ_f", exist_ok=True)
    os.makedirs("phase2_acc_occ_inf", exist_ok=True)

    for single_subject in subjects:
        process_single_subject(single_subject, all_sub_EEG_phase2_f, all_sub_EEG_phase2_inf,
            all_sub_behav_phase2_f, all_sub_behav_phase2_inf)

    print("All subjects have been processed.")