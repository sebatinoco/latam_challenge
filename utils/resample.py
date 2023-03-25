import numpy as np
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def resample(X, y, param_over = False, param_under = False, random_state = 3380):

    '''
    Función que recibe dataframes de entrada y los devuelve resampleados.
    X: dataframe de entrada (X_train)
    y: dataframe de entrada (y_train)
    param_over: factor por el que se hará oversampling
    param_under: factor por el que se hará subsampling
    '''

    
    target_classes = [1] # clase problemática
    
    # Oversampling
    if param_over:
        count_values = y.value_counts()
        dict_over = {key: int(count_values[key] * (1 + param_over)) for key in count_values.keys() if key in target_classes}

        if X.ndim == 1:
            X = np.array(X).reshape(-1, 1)
        oversampler = RandomOverSampler(sampling_strategy = dict_over, random_state = random_state)
        X, y = oversampler.fit_resample(X, y)

    # Subsampling
    if param_under:
        count_values = y.value_counts()
        dict_under = {key: int(count_values[key] * (param_under)) for key in count_values.keys() if key not in target_classes}
        if X.ndim == 1:
            X = np.array(X).reshape(-1, 1)

        undersampler = RandomUnderSampler(sampling_strategy = dict_under, random_state = random_state)
        X, y = undersampler.fit_resample(X, y)

    return X, y