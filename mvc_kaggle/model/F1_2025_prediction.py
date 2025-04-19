import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Função para simular o campeonato de F1 2025
def simular_campeonato_f1_2025():
    # Carrega a base de dados processada
    df = pd.read_csv("mvc_kaggle\model\Data\Processed\DatasetF1processado.csv")

    # Pré-processamento de colunas relevantes
    df = df.dropna(subset=['position', 'grid', 'dob', 'fastestLapSpeed', 'q1'])
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df = df.dropna(subset=['dob'])
    df['age'] = df.apply(lambda row: row['year'] - row['dob'].year, axis=1)
    df['status'] = df['status'].str.lower()
    df['confiavel'] = df['status'].apply(lambda x: 1 if 'finished' in x else 0)

    # Ordena por ano para calcular confiabilidade histórica corretamente
    df = df.sort_values(by=['driverRef', 'year', 'name_circuits'])

    # Inicializa colunas de confiabilidade acumulada
    df['conf_piloto'] = 0
    df['conf_equipe'] = 0

    # Função para calcular confiabilidade acumulada
    def calcular_confiabilidade(historico):
        conf = 0
        confs = []
        for status in historico:
            conf += 1 if 'finished' in status else -1
            confs.append(conf)
        return confs

    # Confiabilidade histórica do piloto
    df['conf_piloto'] = df.groupby('driverRef')['status'].transform(calcular_confiabilidade)

    # Confiabilidade histórica da equipe
    df['conf_equipe'] = df.groupby('name_constructors')['status'].transform(calcular_confiabilidade)

    # Seleciona colunas para o modelo
    features = ['grid', 'fastestLapSpeed', 'confiavel', 'q1', 'age']
    X = df[features]
    y = df['points']

    # Treina o modelo de regressão
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Dados de entrada para a temporada 2025
    corridas_2025 = [
        'Melbourne Grand Prix Circuit',                    # Australian GP
        'Jeddah Street Circuit',                           # Saudi Arabian GP
        'Bahrain International Circuit',                   # Bahrain GP
        'Shanghai International Circuit',                  # Chinese GP
        'Miami International Autodrome',                   # Miami GP
        'Imola - Autodromo Enzo e Dino Ferrari',           # Emilia Romagna GP
        'Circuit de Monaco',                               # Monaco GP
        'Circuit Gilles Villeneuve',                       # Canadian GP
        'Circuit de Barcelona-Catalunya',                  # Spanish GP
        'Red Bull Ring',                                   # Austrian GP
        'Silverstone Circuit',                             # British GP
        'Hungaroring',                                     # Hungarian GP
        'Circuit de Spa-Francorchamps',                    # Belgian GP
        'Circuit Zandvoort',                               # Dutch GP
        'Autodromo Nazionale di Monza',                    # Italian GP
        'Marina Bay Street Circuit',                       # Singapore GP
        'Suzuka International Racing Course',              # Japanese GP
        'Circuit of the Americas (COTA)',                  # United States GP
        'Autódromo Hermanos Rodríguez',                    # Mexican GP
        'Autódromo José Carlos Pace (Interlagos)',         # Brazilian GP
        'Las Vegas Street Circuit',                        # Las Vegas GP
        'Lusail International Circuit',                    # Qatar GP
        'Yas Marina Circuit'                               # Abu Dhabi GP
    ]

    pilotos_2025 = {
        'verstappen': 'Red Bull',
        'tsunoda': 'Red Bull',
        'leclerc': 'Ferrari',
        'hamilton': 'Ferrari',
        'sainz': 'Williams',
        'albon': 'Williams',
        'russell': 'Mercedes',
        'antonelli': 'Mercedes', #não tem 
        'norris': 'McLaren',
        'piastri': 'McLaren',
        'ocon': 'Haas',
        'bearman': 'Haas',
        'alonso': 'Aston Martin',
        'stroll': 'Aston Martin',
        'ocon': 'Alpine',
        'doohan': 'Alpine',
        'lawson': 'AlphaTauri',
        'hadjar': 'AlphaTauri', #não tem 
        'bortoleto': 'Sauber',  #não tem
        'hulkenberg': 'Sauber'
    }

    # Simulação corrida a corrida
    resultados = []

    for corrida in corridas_2025:
        corrida_resultado = []
        for piloto, equipe in pilotos_2025.items():
            historico = df[(df['driverRef'] == piloto.lower()) & (df['name_constructors'] == equipe)]
            if historico.empty:
                historico = df[df['name_constructors'] == equipe]

            media = historico[features].mean()
            if media.isnull().any():
                continue

            pontos_estimados = model.predict([media.values])[0]

            corrida_resultado.append({
                'corrida': corrida,
                'piloto': piloto,
                'equipe': equipe,
                'pontos_previstos': pontos_estimados
            })

        corrida_resultado.sort(key=lambda x: x['pontos_previstos'], reverse=True)
        top10 = corrida_resultado[:10]
        for pos, piloto in enumerate(top10):
            piloto['posicao'] = pos + 1
            if pos == 0:
                piloto['pontos_finais'] = 25
            elif pos == 1:
                piloto['pontos_finais'] = 18
            elif pos == 2:
                piloto['pontos_finais'] = 15
            elif pos == 3:
                piloto['pontos_finais'] = 12
            elif pos == 4:
                piloto['pontos_finais'] = 10
            elif pos == 5:
                piloto['pontos_finais'] = 8
            elif pos == 6:
                piloto['pontos_finais'] = 6
            elif pos == 7:
                piloto['pontos_finais'] = 4
            elif pos == 8:
                piloto['pontos_finais'] = 2
            elif pos == 9:
                piloto['pontos_finais'] = 1

        resultados.extend(top10)

    # DataFrame com resultados
    df_resultados = pd.DataFrame(resultados)

    # Ranking final de pilotos
    ranking_pilotos = df_resultados.groupby('piloto')['pontos_finais'].sum().sort_values(ascending=False).reset_index()
    ranking_pilotos.columns = ['Piloto', 'Pontuacao_Total']

    # Ranking final de equipes
    ranking_equipes = df_resultados.groupby('equipe')['pontos_finais'].sum().sort_values(ascending=False).reset_index()
    ranking_equipes.columns = ['Equipe', 'Pontuacao_Total']

    # Exibindo os resultados
    print("\n===== TOP 10 PILOTOS FINAIS - CAMPEONATO 2025 =====")
    print(ranking_pilotos.head(10).to_string(index=False))

    print("\n===== TOP 10 EQUIPES FINAIS - CAMPEONATO 2025 =====")
    print(ranking_equipes.head(10).to_string(index=False))


# Função main para chamar a simulação
if __name__ == "__main__":
    simular_campeonato_f1_2025()

