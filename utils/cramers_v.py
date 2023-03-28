import pandas as pd
import numpy as np
import scipy.stats as ss

def cramers_v(x, y):
    
    '''
    computes cramer's v score between 2 features
    x: first feature to compute the score (pd.Series)
    y: second feature to compute the score (pd.Series)
    '''
    
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1))/(n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    return np.sqrt(phi2corr/min((kcorr - 1), (rcorr - 1)))