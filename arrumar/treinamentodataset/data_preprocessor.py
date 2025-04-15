import pandas as pd
import numpy as np
from datetime import datetime

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
    Pré-processa os dados e separa as variáveis independentes e dependentes.
    """
    if target not in data.columns:
        raise ValueError(f"A coluna '{target}' não foi encontrada no dataset.")
    
    # Separar variáveis independentes (X) e dependentes (y)
    X = data.drop(columns=[target])
    y = data[target]
    
    return X, y

def aggregate_teams_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados por temporada e equipe (construtora) e retorna as top 5 equipes por temporada.
    """
    # Agregar os dados por temporada e equipe
    equipes_agrupadas = data.groupby(['year', 'name_constructors']).agg({
        'points': 'sum',
        'fastestLapTime': 'mean',
        'fastestLapSpeed': 'mean',
        'fp1_time': 'mean',
        'fp2_time': 'mean',
        'fp3_time': 'mean',
        'quali_time': 'mean',
        'sprint_time': 'mean'
    }).reset_index()

    # Filtrar as top 5 equipes por temporada com base nos pontos
    equipes_top5 = equipes_agrupadas.sort_values(['year', 'points'], ascending=[True, False]).groupby('year').head(5)

    return equipes_top5

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
    Calcula a taxa de crescimento dos pontos para equipes.
    """
    data['taxa_crescimento'] = data.groupby('name_constructors')['points'].pct_change()
    data['taxa_crescimento'] = data['taxa_crescimento'].fillna(0)

    return data

def create_performance_features(equipes_agrupadas: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas características de desempenho com base no dataset.
    """
    corridas_por_ano = data.groupby('year').size().mean()  # Média de corridas por ano
    equipes_agrupadas['media_pontos_por_corrida'] = equipes_agrupadas['points'] / corridas_por_ano

    return equipes_agrupadas

def process_and_analyze_data(file_path: str, target: str) -> pd.DataFrame:
    """
    Função que carrega os dados, faz o pré-processamento, agrupa e analisa as top 5 equipes.
    """
    # Passo 1: Carregar os dados
    data = load_data(file_path)

    # Passo 2: Pré-processar os dados (separar X e y)
    X, y = preprocess_data(data, target)

    # Passo 3: Agrupar dados por equipe e filtrar as top 5 equipes
    equipes_top5 = aggregate_teams_data(data)

    # Passo 4: Adicionar informações de regulamentação
    equipes_top5 = add_regulation_info(data, equipes_top5)

    # Passo 5: Calcular a taxa de crescimento para as equipes
    equipes_top5 = calculate_growth_rate(equipes_top5)

    # Passo 6: Criar características de desempenho
    equipes_top5 = create_performance_features(equipes_top5, data)

    # Retornar os dados das top 5 equipes
    return equipes_top5