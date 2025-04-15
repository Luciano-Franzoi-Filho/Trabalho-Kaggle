import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from data_preprocessor import load_data, preprocess_data

def validate_data(X: pd.DataFrame):
    """
    Valida os dados de entrada para garantir que não contenham valores inválidos (NaN ou nulos).
    """
    if X.isnull().any().any():
        raise ValueError("Os dados de entrada contêm valores NaN ou null.")
    print("Validação dos dados: OK")

def validate_points_column(data: pd.DataFrame):
    """
    Verifica se a coluna 'points' tem valores ausentes ou inválidos.
    """
    if 'points' not in data.columns:
        raise ValueError("A coluna 'points' não foi encontrada no dataset.")
    if data['points'].isnull().any():
        raise ValueError("A coluna 'points' contém valores ausentes.")
    print("A coluna 'points' está sem valores ausentes.")

def create_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """
    Cria o pipeline de pré-processamento e classificação com o modelo fornecido.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Avalia o modelo no conjunto de teste e retorna a acurácia.
    """
    return pipeline.score(X_test, y_test)

def train_model_sklearn(data: pd.DataFrame, target: str):
    """
    Treina múltiplos modelos de classificação e seleciona o melhor.
    """
    X, y = preprocess_data(data, target)
    
    # Corrigir erro de classes únicas no y
    if y.nunique() < 2:
        raise ValueError("O conjunto de dados possui menos de duas classes distintas no target.")
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Ajuste do train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )
    
    candidates = {
        'gradient_boosting': GradientBoostingClassifier(random_state=123),
        'ada_boost': AdaBoostClassifier(random_state=123),
        'lightgbm': LGBMClassifier(random_state=123, device='gpu', n_jobs=-1)
    }
    
    best_model = None
    best_score = -np.inf
    
    for name, model in candidates.items():
        pipeline = create_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        score = evaluate_model(pipeline, X_test, y_test)
        print(f"Modelo: {name} | Acurácia: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = pipeline
    
    return best_model, preprocessor

def predict_2025(modelo_equipes, preprocessor_equipes):
    """
    Faz previsões para a temporada de 2025.
    """
    dados_2025_equipes = pd.DataFrame({
        'year': [2025],
        'media_pontos_por_corrida': [10],
        'taxa_crescimento': [0.05]
    })
    
    dados_2025_equipes_transformed = preprocessor_equipes.transform(dados_2025_equipes)
    equipe_vencedora = modelo_equipes.predict(dados_2025_equipes_transformed)
    print(f"Equipe vencedora prevista para 2025: {equipe_vencedora[0]}")
    
    return equipe_vencedora

def run_pipeline(file_path: str):
    """
    Executa o pipeline de treinamento e previsão.
    """
    data = load_data(file_path)
    validate_points_column(data)
    
    print("Filtrando as top 5 equipes...")
    equipes = data.groupby(['year', 'name_constructors']).apply(
        lambda x: x.nlargest(5, 'points')
    ).reset_index(drop=True)
    
    print("Treinando modelo para equipes...")
    modelo_equipes, preprocessor_equipes = train_model_sklearn(equipes, target='points')
    
    return modelo_equipes, preprocessor_equipes

if __name__ == '__main__':
    file_path = "mvc_kaggle/model/dataset_arrumado/corridas_geral.csv"
    try:
        modelo_equipes, preprocessor_equipes = run_pipeline(file_path)
        predict_2025(modelo_equipes, preprocessor_equipes)
    except Exception as e:
        print(f"Erro no pipeline: {e}")