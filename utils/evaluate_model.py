from sklearn.model_selection import train_test_split
from utils.resample import resample
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import compute_sample_weight

def evaluate_model(df, target, pipeline, param_under = None, param_over = None, fit_weights = False, random_state = 3380, report = True):
    
    '''
    Function that receives a pipeline, trains it, and evaluates it using F1 score.
    df: data on which to execute the pipeline
    target: feature to predict
    pipeline: Pipeline object to be trained
    param_over: Factor by which oversampling will be done (float)
    param_under: Factor by which undersampling will be done (float)
    fit_weights: Indicates whether we want to use the "sample expansion factors". It can improve class imbalance (bool)
    '''

    # stratified holdout
    y = df[target].copy()
    X = df.drop(columns = target).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, shuffle = True, stratify = y, random_state = random_state)

    # resampling
    if (param_under is not None) | (param_over is not None):
        X_train, y_train = resample(X_train, y_train, param_over = param_over, param_under = param_under, random_state = random_state)

    # we can add expansion weights
    if fit_weights:
        sample_weight = compute_sample_weight(class_weight = 'balanced', y = y_train)
        pipeline.fit(X_train, y_train, clf__sample_weight = sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    # train pipeline and obtain predictions
    y_pred = pipeline.predict(X_test)

    # metrics
    
    f1 = f1_score(y_test, y_pred, average = 'macro')
    
    if report:
    
        print(f"F1: {f1:.3f}")
        print(classification_report(y_test, y_pred))