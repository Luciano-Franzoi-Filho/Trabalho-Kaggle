from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def preprocess_data(data, target):
    """
    Preprocessa os dados, aplicando escalonamento e codificação.
    Retorna os dados transformados, o alvo, o preprocessor e os nomes das features.
    """
    # Verificar se a coluna alvo está presente
    if target not in data.columns:
        raise ValueError(f"A coluna alvo '{target}' não está presente no DataFrame.")

    # Separar X e y
    y = data[target]
    X = data.drop(columns=[target])

    print("Colunas disponíveis no DataFrame X:", X.columns.tolist())

    # Identificar colunas com valores NaN
    nan_columns = X.columns[X.isnull().any()].tolist()
    if nan_columns:
        print(f"As seguintes colunas contêm valores NaN: {nan_columns}")
        print("Número de valores NaN por coluna:")
        print(X[nan_columns].isnull().sum())
    else:
        print("Nenhuma coluna contém valores NaN.")

    # Separar colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print("Colunas numéricas:", numeric_features.tolist())
    print("Colunas categóricas:", categorical_features.tolist())

    # Definir transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Criar o ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Aplicar o pré-processador
    try:
        X_preprocessed = preprocessor.fit_transform(X)
        print("Pré-processamento concluído com sucesso.")
    except Exception as e:
        print("Erro durante o pré-processamento:")
        print(e)
        raise

    # Gerar os nomes das features resultantes
    feature_names = []

    if 'num' in dict(preprocessor.named_transformers_):
        feature_names.extend(numeric_features)

    if 'cat' in dict(preprocessor.named_transformers_):
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)

    return X_preprocessed, y, preprocessor, feature_names
