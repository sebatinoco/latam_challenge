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
    Función que recibe un pipeline y lo entrena.
    X_train: Matriz de features de entrenamiento (np.array)
    y_train: Vector objetivo a predecir (np.array)
    pipeline: Objeto Pipeline a ser entrenado (sklearn.Pipeline)
    param_over: Factor por el que se hará oversampling (float)
    param_under: Factor por el que se hará subsampling (float)
    fit_weights: Indica si queremos usar los "factores de expansión" de la muestra. Puede mejorar el desbalance de clases (bool)
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

    # podemos especificar under y over sampling
    if (param_under is not None) | (param_over is not None):
        X_train, y_train = resample(X_train, y_train, param_over = param_over, param_under = param_under, random_state = random_state)

    pipeline = deepcopy(base_pipeline)
    pipeline.steps.append(('clf', LGBMClassifier(random_state = random_state, **lgbm_params)))

    # entrenamos, podemos agregar factores de expansión
    if fit_weights:
        sample_weight = compute_sample_weight(class_weight = 'balanced', y = y_train)
        pipeline.fit(X_train, y_train, clf__sample_weight = sample_weight)
    else:
        pipeline.fit(X_train, y_train)


    cv = KFold(n_splits = 10, shuffle = True, random_state = random_state)
    scores = cross_val_score(pipeline, X_train, y_train , scoring = 'f1_macro', cv = cv)
    f1_avg = np.mean(scores)
    
    return f1_avg