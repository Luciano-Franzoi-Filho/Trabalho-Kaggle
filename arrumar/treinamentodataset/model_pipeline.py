import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

# -------------------- Funções de Pré-Processamento --------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.
    """
    data = pd.read_csv(file_path)
    
    if 'points' not in data.columns:
        raise ValueError("A coluna 'points' não foi encontrada no dataset.")
    
    return data

def preprocess_data(data: pd.DataFrame, target: str) -> tuple:
    """
    Pré-processa os dados e separa as variáveis independentes e dependentes, incluindo
    a coluna 'status' para verificar a confiabilidade do carro.
    """
    print("Iniciando o pré-processamento dos dados...")

    if target not in data.columns:
        raise ValueError(f"A coluna '{target}' não foi encontrada no dataset.")
    
    # Verificar se a coluna 'dob' (data de nascimento) existe no DataFrame
    if 'dob' not in data.columns:
        raise ValueError("A coluna 'dob' (data de nascimento) não foi encontrada no dataset.")
    
    # Verificar se a coluna 'status' existe no DataFrame
    if 'status' not in data.columns:
        raise ValueError("A coluna 'status' não foi encontrada no dataset.")
    
    # Calcular a idade do piloto a partir da coluna 'dob'
    data['idade'] = data['dob'].apply(calcular_idade)
    
    # Criar a coluna 'carro_confiavel' baseada no 'status' do carro
    # Se o status for 'Running', consideramos o carro confiável (1), caso contrário, não (0)
    data['carro_confiavel'] = data['status'].apply(lambda x: 1 if x == 'Running' else 0)
    
    # Criar novas variáveis: Melhor tempo de qualificação (menor tempo entre q1, q2, q3)
    data['melhor_temporada'] = data[['q1', 'q2', 'q3']].min(axis=1)  # Menor tempo é o melhor
    data['carro_rapido'] = data['fastestLapTime']  # Melhor tempo de volta do carro
    data['fastestLapSpeed'] = data['fastestLapSpeed']  # Velocidade máxima da volta
    
    # Variáveis de posição: Posição de largada e posição final de chegada
    data['posicao_largada'] = data['grid']
    data['posicao_chegada'] = data['position']
    
    # Substituir NaN por 0 nas colunas necessárias
    data.fillna(0, inplace=True)  # Substitui todos os NaN do DataFrame por 0

    # Separar variáveis independentes (X) e dependentes (y)
    X = data.drop(columns=[target, 'dob', 'time'])  # Remover colunas não úteis
    y = data[target]
    
    print(f"Dimensões de X após o pré-processamento: {X.shape}")  # Verificar número de features
    
    return X, y

def calcular_idade(data_nascimento: str) -> int:
    """
    Calcula a idade do piloto a partir da data de nascimento (no formato 'YYYY-MM-DD').
    """
    # Convertendo a string 'dob' para o tipo datetime
    nascimento = datetime.strptime(data_nascimento, '%Y-%m-%d')
    
    # Calculando a diferença entre a data atual e a data de nascimento
    idade = datetime.now().year - nascimento.year
    
    # Ajustando a idade caso o aniversário do piloto ainda não tenha ocorrido no ano
    if datetime.now().month < nascimento.month or (datetime.now().month == nascimento.month and datetime.now().day < nascimento.day):
        idade -= 1
    
    return idade

def add_idade_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona a coluna 'idade' ao DataFrame a partir da coluna 'dob'.
    """
    data['idade'] = data['dob'].apply(calcular_idade)
    return data

def fill_na_with_zero(data: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui todos os valores NaN no DataFrame por 0.
    """
    return data.fillna(0)

def aggregate_teams_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados por temporada e equipe (construtora), mantendo a coluna original 'points' 
    e criando uma nova coluna 'total_points' com a soma dos pontos por equipe e ano.
    """
    # Criar a coluna 'total_points' somando os pontos por ano e equipe
    equipes_agrupadas = data.groupby(['year', 'name_constructors']).agg({
        'points': 'sum'
    }).rename(columns={'points': 'total_points'}).reset_index()

    # Manter a coluna original 'points' sem somar, pegando o valor original do dataset
    equipes_agrupadas = equipes_agrupadas.merge(
        data[['year', 'name_constructors', 'points', 'dob', 'status',
               'q1', 'q2', 'q3','fastestLapTime','fastestLapSpeed',
                 'grid', 'position', 'time']], 
        on=['year', 'name_constructors'],
        how='left'
    ).drop_duplicates()

    return equipes_agrupadas

def add_regulation_info(data: pd.DataFrame, equipes_agrupadas: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona informações de regulamentação aos dados agrupados de equipes.
    """
    equipes_agrupadas = equipes_agrupadas.merge(
        data[['year', 'regulamento']].drop_duplicates(),
        on='year',
        how='left'
    )

    return equipes_agrupadas

def calculate_growth_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a taxa de crescimento dos pontos para equipes, evitando valores infinitos.
    """
    # Calcular a taxa de crescimento da coluna 'points'
    data['taxa_crescimento'] = data.groupby('name_constructors')['points'].pct_change()
    
    # Substituir valores infinitos ou NaN por zero
    data['taxa_crescimento'] = data['taxa_crescimento'].replace([np.inf, -np.inf], 0)
    data['taxa_crescimento'] = data['taxa_crescimento'].fillna(0)

    return data

def create_performance_features(equipes_agrupadas: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas características de desempenho com base no dataset.
    """
    corridas_por_ano = data.groupby('year').size().mean()  # Média de corridas por ano
    equipes_agrupadas['media_pontos_por_corrida'] = equipes_agrupadas['points'] / corridas_por_ano
    
    # Substituir qualquer NaN gerado pela divisão por 0 (caso haja corridas por ano igual a 0)
    equipes_agrupadas['media_pontos_por_corrida'].fillna(0, inplace=True)

    return equipes_agrupadas

# -------------------- Funções de Treinamento e Previsão --------------------

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
    Treina múltiplos modelos de regressão e seleciona o melhor, considerando novas variáveis.
    """
    X, y = preprocess_data(data, target)
    
    # Verificar novamente as dimensões de X após o pré-processamento
    print(f"Dimensões de X após o pré-processamento em train_model_sklearn: {X.shape}")

    # Separando features numéricas e categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Verificar as dimensões antes de aplicar qualquer transformação
    print(f"Dimensões de X (numéricas e categóricas separadas): {numeric_features.shape[0]} numéricas, {categorical_features.shape[0]} categóricas")

    # Definindo os transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    # Verificar as dimensões de X_train e X_test após a divisão
    print(f"Dimensões de X_train: {X_train.shape}")
    print(f"Dimensões de X_test: {X_test.shape}")
    
    # Modelos a serem testados (REGRESSÃO)
    candidates = {
        'gradient_boosting': GradientBoostingRegressor(random_state=123),
        'random_forest': RandomForestRegressor(random_state=123),
        'lightgbm': LGBMRegressor(random_state=123, n_jobs=-1)
    }
    
    best_model = None
    best_score = np.inf  # Para regressão, o melhor é o menor erro

    for name, model in candidates.items():
        pipeline = create_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        print(f"Modelo: {name} | Erro absoluto médio: {score:.4f}")
        if score < best_score:
            best_score = score
            best_model = pipeline
    
    print(f"Melhor modelo: {best_model.steps[-1][1].__class__.__name__} | Erro: {best_score:.4f}")
    return best_model, preprocessor

def predict_2025(modelo_equipes, preprocessor_equipes):
    """
    Faz previsões para a temporada de 2025, considerando as variáveis de desempenho do carro.
    """
    # print("modelo_equipes:")
    # print(modelo_equipes)
    # print("preprocessor_equipes:")
    # print(preprocessor_equipes)

    # Criar o DataFrame com todas as colunas esperadas pelo preprocessor
    dados_2025_equipes = pd.DataFrame({
        'year': [2025, 2025],
        'media_pontos_por_corrida': [25, 18],  # Pontos de Oscar Piastri e Lando Norris
        'taxa_crescimento': [0.05, 0.05],
        'carro_confiavel': [1, 1],
        'melhor_temporada': [150000, 150000],  # Fictício, ajustar conforme necessário
        'fastestLapSpeed': [205.439, 205.581],  # Velocidade média da volta mais rápida
        'posicao_largada': [1, 2],  # Piastri largou em 1º, Norris em 2º no GP da China
        'posicao_chegada': [1, 2],  # Piastri venceu, Norris foi 2º no GP da China
        'idade': [23, 25],  # Idade de Oscar Piastri em 2025
        'total_points': [25, 18],  # Pontos acumulados após o GP da China
        'regulamento': ['default', 'default'],
        'position': [1, 2],  # Posição final na corrida
        'fastestLapTime': [90641, 90793],  # Tempo da volta mais rápida em milissegundos (1:30.641 e 1:30.793)
        'name_constructors': ['McLaren', 'McLaren'],
        'q1': [91591, 91324],  # Tempos de qualificação (Q1) em milissegundos
        'q2': [91200, 90787],  # Tempos de qualificação (Q2) em milissegundos
        'q3': [90641, 90793],  # Tempos de qualificação (Q3) em milissegundos
        'carro_rapido': [0, 0],  # Sem dados específicos
        'status': ['Running', 'Running'],
        'grid': [1, 2]  # Posições de largada
    })

    print(f"Colunas em dados_2025_equipes: {dados_2025_equipes.shape[1]}")

    # Garantir que as colunas estejam na mesma ordem e formato esperado pelo preprocessor
    colunas_esperadas = preprocessor_equipes.feature_names_in_
    dados_2025_equipes = dados_2025_equipes[colunas_esperadas]

    print(f"Colunas esperadas pelo preprocessor: {preprocessor_equipes.feature_names_in_}")

    # Transformar os dados usando o preprocessor
    dados_2025_equipes_transformed = preprocessor_equipes.transform(dados_2025_equipes)
    
    # Fazer a previsão com o modelo
    pontuacao_prevista = modelo_equipes.predict(dados_2025_equipes_transformed)
    print(f"Pontuação prevista para 2025: {pontuacao_prevista[0]}")

    return pontuacao_prevista

def run_pipeline(file_path: str):
    """
    Executa o pipeline de treinamento e previsão.
    """
    # pd.set_option('display.max_columns', None)  # Exibir todas as colunas
    # pd.set_option('display.width', None)        # Permitir a largura completa para exibição
    # pd.set_option('display.max_colwidth', None) # Não limitar o comprimento das colunas

    data = load_data(file_path)
    validate_points_column(data)
    # print(data)
    
    print("Filtrando as top 5 equipes...")
    equipes = aggregate_teams_data(data)
    # print(equipes)
    equipes = add_regulation_info(data, equipes)
    # print(equipes)
    equipes = calculate_growth_rate(equipes)
    # print(equipes)
    equipes = create_performance_features(equipes, data)
    # print(equipes)
    
    print("Treinando modelo para equipes...")
    modelo_equipes, preprocessor_equipes = train_model_sklearn(equipes, target='points')
    
    return modelo_equipes, preprocessor_equipes