import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(file_path: str):
    """
    Carrega e pré-processa os dados:
    - Codifica variáveis categóricas
    - Divide os dados em variáveis independentes (X) e dependentes (y)
    """
    # Carregar os dados
    df = load_data(file_path)

    # Codificar variáveis categóricas
    label_encoders = {}
    for column in ['status', 'name_constructors', 'driverRef']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Selecionar variáveis independentes e dependentes
    X = df[['year', 'grid', 'points', 'status']]
    y_pilotos = df['driverRef']  # Previsão dos pilotos
    y_equipes = df['name_constructors']  # Previsão das equipes

    return X, y_pilotos, y_equipes, label_encoders

def train_model(file_path: str, output_model_path: str):
    """
    Treina modelos de machine learning para prever os 5 primeiros pilotos e equipes.
    """
    # Pré-processar os dados
    X, y_pilotos, y_equipes, label_encoders = preprocess_data(file_path)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train_pilotos, y_test_pilotos = train_test_split(X, y_pilotos, test_size=0.3, random_state=42)
    _, _, y_train_equipes, y_test_equipes = train_test_split(X, y_equipes, test_size=0.3, random_state=42)

    # Criar e treinar o modelo para pilotos
    model_pilotos = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model_pilotos.fit(X_train, y_train_pilotos)

    # Criar e treinar o modelo para equipes
    model_equipes = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model_equipes.fit(X_train, y_train_equipes)

    # Avaliar o desempenho no conjunto de teste para pilotos
    y_pred_pilotos = model_pilotos.predict(X_test)
    accuracy_pilotos = accuracy_score(y_test_pilotos, y_pred_pilotos)
    print(f"Accuracy (Pilotos): {accuracy_pilotos}")
    print("Classification Report (Pilotos):")
    print(classification_report(y_test_pilotos, y_pred_pilotos))

    # Avaliar o desempenho no conjunto de teste para equipes
    y_pred_equipes = model_equipes.predict(X_test)
    accuracy_equipes = accuracy_score(y_test_equipes, y_pred_equipes)
    print(f"Accuracy (Equipes): {accuracy_equipes}")
    print("Classification Report (Equipes):")
    print(classification_report(y_test_equipes, y_pred_equipes))

    # Salvar os modelos treinados
    joblib.dump((model_pilotos, model_equipes, label_encoders), output_model_path)

    return model_pilotos, model_equipes, label_encoders, accuracy_pilotos, accuracy_equipes

def predict_top_5(model_path: str, input_data: dict):
    """
    Faz previsões sobre os 5 primeiros pilotos e equipes com base em novos dados.
    """
    # Carregar os modelos treinados e os codificadores
    model_pilotos, model_equipes, label_encoders = joblib.load(model_path)

    # Preparar os dados de entrada
    input_df = pd.DataFrame([input_data])
    input_df['status'] = label_encoders['status'].transform(input_df['status'])

    # Fazer as previsões
    predicted_pilotos = model_pilotos.predict(input_df)
    predicted_equipes = model_equipes.predict(input_df)

    # Decodificar os resultados
    top_5_pilotos = label_encoders['driverRef'].inverse_transform(predicted_pilotos)[:5]
    top_5_equipes = label_encoders['name_constructors'].inverse_transform(predicted_equipes)[:5]

    return top_5_pilotos, top_5_equipes

if __name__ == "__main__":
    # Caminho do arquivo de dados
    file_path = 'mvc_kaggle/model/dataset_arrumado/corridas_geral.csv'

    # Caminho para salvar os modelos treinados
    output_model_path = 'mvc_kaggle/model/treinamentodataset/team_driver_model.pkl'

    # Treinar os modelos
    model_pilotos, model_equipes, label_encoders, accuracy_pilotos, accuracy_equipes = train_model(file_path, output_model_path)
    print(f"Modelos treinados com sucesso! Accuracy (Pilotos): {accuracy_pilotos}, Accuracy (Equipes): {accuracy_equipes}")

    # Fazer uma previsão de exemplo
    input_data = {
        'year': 2024,
        'grid': 1,
        'points': 25,
        'status': 'Finished'
    }
    top_5_pilotos, top_5_equipes = predict_top_5(output_model_path, input_data)
    print(f"Os 5 primeiros pilotos previstos são: {top_5_pilotos}")
    print(f"As 5 primeiras equipes previstas são: {top_5_equipes}")