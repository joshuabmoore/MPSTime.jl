import json
import numpy as np
import matplotlib.pyplot as plt
from pypots.optim import Adam
from pypots.imputation import CSDI, BRITS
from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
import pickle
import sys
# sys.path.append("Interpolation/Imputation_Algs")
# from cdrec.python.recovery import centroid_recovery as CDrec
# set_random_seed(1234)
# # check that GPU acceleration is enabled
# import torch
# torch.cuda.device_count()
# print(f"GPU: {torch.cuda.get_device_name()}")
# print(f"CUDA ENABLED: {torch.cuda.is_available()}")

# load the folds directly, no idxs required
with open("c4_folds_python.json", "r") as f:
    c4_folds = json.load(f)
print("Successfully loaded c4 folds.")

X_train_original = np.array(c4_folds[0][0][0])
X_test_original = np.array(c4_folds[0][1][0])

# load the python idx adjusted window locations
with open("kepler_windows_python_idx.json", "r") as f:
    window_idxs = json.load(f)
print("Successfully loaded window idxs.")

def evaluate_folds_csdi(folds, window_idxs, model):
    fold_scores = dict()
    for fold in range(0, 1):
        instance_scores = dict()
        print(f"Evaluating fold {fold}/{len(folds)-1}...")
        for instance in range(0, 1):
            X_train_fold_instance = np.array(folds[fold][0][instance]).reshape(-1, 100, 1)
            X_test_fold_instance = np.array(folds[fold][1][instance]).reshape(-1, 100, 1)
            print(f"Training CSDI on fold {fold}, instance {instance}...")
            model.fit(train_set={'X':X_train_fold_instance})
            print("Finished training!")
            # loop over % missing
            percent_missing_score = dict()
            for pm in window_idxs:
                print(f"Imputing {pm}% missing data over {len(window_idxs[pm])} windows...")
                per_window_scores = dict()
                for (idx, widx) in enumerate(window_idxs[pm]):
                    X_test_fold_instance_corrupted = X_test_fold_instance.copy()
                    X_test_fold_instance_corrupted[:, widx] = np.nan
                    mask = np.isnan(X_test_fold_instance_corrupted)
                    csdi_imputed = model.impute(test_set={'X': X_test_fold_instance_corrupted}).squeeze(axis=1)
                    # adjust to normalise by the time series scale
                    errs = np.array([calc_mae(csdi_imputed[i], X_test_fold_instance[i], mask[i]) for i in range(0, X_test_fold_instance.shape[0])])
                    # merge data and compute min/max
                    merged_data = np.vstack([X_train_fold_instance, X_test_fold_instance])
                    max_val = np.max(np.max(merged_data))
                    min_val = np.min(np.min(merged_data))
                    ra = max_val - min_val
                    errs_normalised = errs/ra
                    per_window_scores[idx] = errs_normalised
                percent_missing_score[pm] = per_window_scores
            instance_scores[instance] = percent_missing_score
        fold_scores[fold] = instance_scores
    return fold_scores

csdi = CSDI(
    n_steps=100,
    n_features=1, # univariate time series, so num features is equal to one
    n_layers=6,
    n_heads=2,
    n_channels=128,
    d_time_embedding=64,
    d_feature_embedding=32,
    d_diffusion_embedding=128,
    target_strategy="random",
    n_diffusion_steps=50,
    batch_size=32,
    epochs=1,
    patience=None,
    optimizer=Adam(lr=1e-3),
    num_workers=0,
    device=None,
    model_saving_strategy=None
)

fold_scores_csdi = evaluate_folds_csdi(c4_folds, window_idxs, csdi)
with open("kep_c4_trial_csdi_results.pkl", "wb") as f:
    pickle.dump(fold_scores_csdi, f)