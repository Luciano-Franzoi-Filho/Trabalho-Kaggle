import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import numpy as np

def train_model(data: pd.DataFrame, target: str):
    """
    Treina o modelo usando scikit-learn.
    """
    X = data.drop(columns=[target])
    y = data[target]
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    candidates = {
        'gradient_boosting': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(random_state=123))
            ]),
            'params': {
                'clf__n_estimators': [50, 100, 200],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7]
            }
        },
        'ada_boost': {
            'pipeline': Pipeline([
                ('clf', AdaBoostClassifier(random_state=123))
            ]),
            'params': {
                'clf__n_estimators': [50, 100, 200],
                'clf__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'lightgbm': {
            'pipeline': Pipeline([
                ('clf', LGBMClassifier(random_state=123))
            ]),
            'params': {
                'clf__n_estimators': [50, 100, 200],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [-1, 5, 10]
            }
        }
    }
    
    best_score = -np.inf
    best_model = None
    
    for name, candidate in candidates.items():
        grid = GridSearchCV(candidate['pipeline'], candidate['params'], cv=5, scoring='accuracy', n_jobs=1)
        grid.fit(X, y)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_

    return best_model, le