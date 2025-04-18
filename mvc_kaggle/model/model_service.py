import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(data: pd.DataFrame, target: str):
    """
    Treina modelos de regressão com GridSearchCV e gera gráfico de comparação de performance.
    Também realiza validação cruzada e exibe o R² médio.
    """
    X = data.drop(columns=[target])
    y = data[target]
    
    # Encode se o target for categórico (não deve ser o caso aqui)
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    # Modelos de regressão
    candidates = {
        'Gradient Boosting': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('reg', GradientBoostingRegressor(random_state=123))
            ]),
            'params': {
                'reg__n_estimators': [100],
                'reg__learning_rate': [0.1],
                'reg__max_depth': [3]
            }
        },
        'AdaBoost': {
            'pipeline': Pipeline([
                ('reg', AdaBoostRegressor(random_state=123))
            ]),
            'params': {
                'reg__n_estimators': [100],
                'reg__learning_rate': [0.1]
            }
        },
        'LightGBM': {
            'pipeline': Pipeline([
                ('reg', LGBMRegressor(random_state=123))
            ]),
            'params': {
                'reg__n_estimators': [100],
                'reg__learning_rate': [0.1],
                'reg__max_depth': [5]
            }
        },
        'Random Forest': {
            'pipeline': Pipeline([
                ('reg', RandomForestRegressor(random_state=123))
            ]),
            'params': {
                'reg__n_estimators': [100],
                'reg__max_depth': [10]
            }
        },
        'SVM': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('reg', SVR())
            ]),
            'params': {
                'reg__C': [1],
                'reg__kernel': ['rbf'],
                'reg__gamma': ['scale']
            }
        }
    }

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    scores = {}
    best_score = -np.inf
    best_model = None
    models = {}

    # Treinamento + avaliação
    for name, candidate in candidates.items():
        print(f"\nTreinando modelo: {name}")
        grid = GridSearchCV(candidate['pipeline'], candidate['params'], cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        models[name] = grid.best_estimator_
        y_pred = grid.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        scores[name] = r2

        print(f"{name} - MAE: {mae:.2f} | RMSE: {rmse:.2f} | R² Teste: {r2:.3f}")

        # Validação cruzada
        cv_scores = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring='r2')
        print(f"{name} - R² Validação Cruzada (média): {cv_scores.mean():.3f}")

        if r2 > best_score:
            best_score = r2
            best_model = grid.best_estimator_

    # Mostrar performance no treino
    print("\nR² no Conjunto de Treinamento:")
    for name, model in models.items():
        train_r2 = model.score(X_train, y_train)
        print(f"{name} - R² Treino: {train_r2:.4f}")

    # Geração do gráfico comparativo de R²
    plt.figure(figsize=(10, 6))
    model_names = list(scores.keys())
    r2_scores = list(scores.values())
    bars = plt.bar(model_names, r2_scores, color=['skyblue', 'orange', 'green', 'purple', 'red'])

    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.2f}', ha='center', va='bottom')

    plt.ylabel('R² Score (Teste)')
    plt.title('Comparação de Modelos de Regressão')
    plt.ylim(0, 1)

    # Criar pasta para relatórios
    output_dir = 'mvc_kaggle/view/relatorios'
    os.makedirs(output_dir, exist_ok=True)

    # Salvar gráfico
    output_path = os.path.join(output_dir, 'comparacao_modelos_regressao.png')
    plt.savefig(output_path)
    plt.close()
    print(f"\nGráfico de desempenho salvo em: {output_path}")

    return best_model, le
