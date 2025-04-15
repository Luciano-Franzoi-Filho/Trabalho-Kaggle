from model.data_repository import load_data
from model.data_preprocessor import preprocess_data
from model.model_service import train_model
from view.view_service import display_results
import shap
import pandas as pd

def run_pipeline(file_path: str, frac: float):
    """
    Executa o pipeline de treinamento e análise do modelo.
    
    Args:
        file_path (str): Caminho para o arquivo de dados.
        frac (float): Fração dos dados a ser utilizada no pipeline (ex.: 0.01 para 1% dos dados).
    """
    target = 'points'

    # Load raw data
    data = load_data(file_path)

    # Verificar se a coluna alvo está presente
    if target not in data.columns:
        raise ValueError(f"A coluna alvo '{target}' não está presente no DataFrame.")

    # Limitar o tamanho da base de dados usando a fração especificada
    if frac > 0 and frac <= 1:
        data = data.sample(frac=frac, random_state=42)  # Seleciona uma fração aleatória dos dados
        print(f"Base de dados reduzida para {len(data)} linhas (fração: {frac}).")
    else:
        raise ValueError("O argumento 'frac' deve estar entre 0 e 1.")

    # Preprocess data
    X_preprocessed, y, preprocessor = preprocess_data(data, target)
    print("Pré-processamento concluído.")

    # Convert sparse matrix to dense if necessary
    if hasattr(X_preprocessed, "toarray"):
        X_preprocessed = X_preprocessed.toarray()

    # Create a new DataFrame with preprocessed features
    X_df = pd.DataFrame(X_preprocessed)
    print("Colunas disponíveis no DataFrame pré-processado (X_df):", X_df.columns.tolist())
    data_preprocessed = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
    print("Colunas disponíveis no DataFrame final (data_preprocessed):", data_preprocessed.columns.tolist())

    # Verificar novamente se a coluna alvo está presente
    if target not in data_preprocessed.columns:
        raise ValueError(f"A coluna alvo '{target}' não está presente no DataFrame final.")

    # Train the model
    model, le = train_model(data_preprocessed, target)
    
    # Analyze features
    X = data_preprocessed.drop(columns=[target])
    explainer = shap.Explainer(model.predict, X)
    shap_summary = explainer(X)
    
    # Display results
    display_results(shap_summary)