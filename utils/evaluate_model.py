from sklearn.model_selection import train_test_split
from utils.resample import resample
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import compute_sample_weight

def evaluate_model(df, target, pipeline, param_under = None, param_over = None, fit_weights = False, random_state = 3380, report = True):
    '''
    Función que recibe un pipeline y lo entrena.
    df: datos sobre los que ejecutar el pipeline
    target: feature a predecir
    pipeline: Objeto Pipeline a ser entrenado
    param_over: Factor por el que se hará oversampling (float)
    param_under: Factor por el que se hará subsampling (float)
    fit_weights: Indica si queremos usar los "factores de expansión" de la muestra. Puede mejorar el desbalance de clases (bool)
    '''

    # dividimos datos usando estratificación
    y = df[target].copy()
    X = df.drop(columns = target).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, shuffle = True, stratify = y, random_state = random_state)

    # podemos especificar under y over sampling
    if (param_under is not None) | (param_over is not None):
        X_train, y_train = resample(X_train, y_train, param_over = param_over, param_under = param_under, random_state = random_state)

    # entrenamos, podemos agregar factores de expansión
    if fit_weights:
        sample_weight = compute_sample_weight(class_weight = 'balanced', y = y_train)
        pipeline.fit(X_train, y_train, clf__sample_weight = sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    # entrenamos pipeline y obtenemos predicciones
    y_pred = pipeline.predict(X_test)

    # metrics
    
    f1 = f1_score(y_test, y_pred, average = 'macro')
    
    if report:
    
        print(f"F1: {f1:.3f}")
        print(classification_report(y_test, y_pred))