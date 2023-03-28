import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold, cross_val_score
from utils.resample import resample
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

def optimize_xgboost(X_train, y_train, base_pipeline, random_state = 3380,
                    param_under = None, param_over = None, fit_weights = False,
                   max_depth = 6, learning_rate = 0.3, n_estimators = 50, min_child_weight = 1, 
                   gamma = 0, subsample = 1, colsample_bytree = 1, reg_alpha = 0, reg_lambda = 1):
    '''
    Function that receives a pipeline and trains it using XGBoost.
    X_train: Training features matrix (np.array)
    y_train: Target vector to predict (np.array)
    pipeline: Pipeline object to be trained (sklearn.Pipeline)
    param_over: Factor by which oversampling will be done (float)
    param_under: Factor by which undersampling will be done (float)
    fit_weights: Indicates whether we want to use the "sample expansion factors". It can improve class imbalance (bool)
    max_depth: Maximum tree depth for base learners. (int)
    learning_rate: Boosting learning rate (xgb's "eta") (float)
    n_estimators: Number of boosting rounds (int)
    min_child_weight: Minimum sum of instance weight (hessian) needed in a child (float)
    gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree (float)
    subsample: Subsample ratio of the training instances (float)
    colsample_bytree: subsample ratio of columns when constructing each tree (float)
    reg_alpha: L2 regularization term on weights (float)
    reg_lambda: L2 regularization term on weights (float)
    '''

    xgboost_params = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        }

    # resampling
    if (param_under is not None) | (param_over is not None):
        X_train, y_train = resample(X_train, y_train, param_over = param_over, param_under = param_under, random_state = random_state)

    pipeline = deepcopy(base_pipeline)
    pipeline.steps.append(('clf', XGBClassifier(random_state = random_state, **xgboost_params)))

    # we can add expansion weights
    if fit_weights:
        sample_weight = compute_sample_weight(class_weight = 'balanced', y = y_train)
        pipeline.fit(X_train, y_train, clf__sample_weight = sample_weight)
    else:
        pipeline.fit(X_train, y_train)


    # evaluate using KFold CV
    cv = KFold(n_splits = 10, shuffle = True, random_state = random_state)
    scores = cross_val_score(pipeline, X_train, y_train , scoring = 'f1_macro', cv = cv)
    f1_avg = np.mean(scores)
    
    return f1_avg