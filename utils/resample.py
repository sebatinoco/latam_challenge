import numpy as np
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def resample(X, y, param_over = False, param_under = False, random_state = 3380):

    '''
    Function that receives input dataframes and returns resampled dataframes.
    X: input dataframe (X_train)
    y: input dataframe (y_train)
    param_over: factor by which oversampling will be done
    param_under: factor by which undersampling will be done
    '''

    
    target_classes = [1] # minority class
    
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