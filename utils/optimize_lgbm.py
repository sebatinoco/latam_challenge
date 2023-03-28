import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold, cross_val_score
from utils.resample import resample
from sklearn.utils import compute_sample_weight
from lightgbm import LGBMClassifier

def optimize_lgbm(X_train, y_train, base_pipeline, random_state = 3380,
                  param_under = None, param_over = None, fit_weights = False,
                  max_depth = 6, learning_rate = 0.3, n_estimators = 50, num_leaves = 31,
                  min_child_samples = 20, reg_alpha = 0, reg_lambda = 0, min_split_gain = 0,
                  subsample = 1.0, subsample_freq = 0, colsample_bytree = 1.0):
    
    '''
    Function that receives a pipeline and trains it using LightGBM.
    X_train: Training features matrix (np.array)
    y_train: Target vector to predict (np.array)
    pipeline: Pipeline object to be trained (sklearn.Pipeline)
    param_over: Factor by which oversampling will be done (float)
    param_under: Factor by which undersampling will be done (float)
    fit_weights: Indicates whether we want to use the "sample expansion factors". It can improve class imbalance (bool)
    max_depth: limit the max depth for tree model (int)
    learning_rate: model learning rate (float)
    n_estimators: number of boosting iterations (int)
    num_leaves: max number of leaves in one tree (int)
    min_child_samples: minimal number of data in one leaf (int)
    reg_alpha: L1 regularization (float)
    reg_lambda: L2 regularization (float)
    min_split_gain: the minimal gain to perform split (float)
    subsample: randomly select part of data without resampling (float)
    subsample_freq: frequency for bagging (int)
    colsample_bytree: select a subset of features on each iteration (float)
    '''

    lgbm_params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'min_split_gain': min_split_gain,
        'subsample': subsample,
        'subsample_freq': subsample_freq,
        'colsample_bytree': colsample_bytree,
        }

    # resampling
    if (param_under is not None) | (param_over is not None):
        X_train, y_train = resample(X_train, y_train, param_over = param_over, param_under = param_under, random_state = random_state)

    pipeline = deepcopy(base_pipeline)
    pipeline.steps.append(('clf', LGBMClassifier(random_state = random_state, **lgbm_params)))

    # we can add expansion weights
    if fit_weights:
        sample_weight = compute_sample_weight(class_weight = 'balanced', y = y_train)
        pipeline.fit(X_train, y_train, clf__sample_weight = sample_weight)
    else:
        pipeline.fit(X_train, y_train)


    # evaluate using KFold CV
    cv = KFold(n_splits = 10, shuffle = True, random_state = random_state)
    scores = cross_val_score(pipeline, X_train, y_train , scoring = 'f1_macro', cv = cv)
    f1_avg = np.mean(scores)
    
    return f1_avg