import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mvc_kaggle/logs/logs.txt"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("F1_Data_Processing")

logger = setup_logger()

def baixar_dataset(api, dataset="rohanrao/formula-1-world-championship-1950-2020", destino="mvc_kaggle/model/Data/Raw"):
    try:
        os.makedirs(destino, exist_ok=True)
        logger.info(f"Baixando dataset {dataset} para {destino}...")
        api.dataset_download_files(dataset, path=destino, unzip=True)
        logger.info("Download concluído e arquivos extraídos com sucesso!")

        arquivos = {arq.split('.')[0]: pd.read_csv(os.path.join(destino, arq)) for arq in os.listdir(destino) if arq.endswith('.csv')}
        logger.info("Arquivos carregados com sucesso!")

        logger.info("Iniciando o processamento dos datasets...")
        dataset_final = ajustar_datasets(arquivos)

        salvar_dataset(dataset_final, destino_final="mvc_kaggle/model/Data/Processed")
        logger.info("Processamento concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao processar o dataset: {e}")

def convert_time_to_milliseconds(time_str):
    """
    Converte um tempo no formato MM:SS.mmm para milissegundos.
    """
    try:
        if isinstance(time_str, str) and ':' in time_str:
            minutes, seconds = time_str.split(':')
            seconds, milliseconds = seconds.split('.')
            total_milliseconds = (int(minutes) * 60 * 1000) + (int(seconds) * 1000) + int(milliseconds)
            return total_milliseconds
        return None  # Retorna None para valores inválidos
    except ValueError:
        return None  # Retorna None para valores inválidos

def ajustar_datasets(arquivos):
    try:
        required_files = ['races', 'results', 'drivers', 'constructors', 'circuits', 'status', 'qualifying']
        for file in required_files:
            if file not in arquivos:
                raise FileNotFoundError(f"Arquivo necessário '{file}.csv' não encontrado.")

        # Substituir '\N' por NaN em todos os DataFrames
        for key in arquivos:
            arquivos[key].replace(r'\\N', pd.NA, inplace=True, regex=True)

        # Converter colunas de IDs para numéricas
        for df_name in required_files:
            df = arquivos[df_name]
            for col in ['raceId', 'driverId', 'constructorId', 'circuitId', 'statusId']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Relacionar tabelas
        races = arquivos['races'].merge(
            arquivos['circuits'][['circuitId', 'name', 'country']],
            on="circuitId", how="left", suffixes=("_gp", "_circuit")
        )
        races.rename(columns={'name_circuit': 'name_circuits'}, inplace=True)

        # Adicionar tempos de treino, classificação e ano da tabela 'races' ao 'results'
        results = arquivos['results'].merge(
            races[['raceId', 'year', 'name_circuits', 'country', 'fp1_time', 'fp2_time', 'fp3_time', 'quali_time', 'sprint_time']],
            on="raceId", how="left"
        )

        # Relacionar com drivers para obter o número do piloto
        if 'number' in arquivos['drivers'].columns:
            results = results.merge(
                arquivos['drivers'][['driverId', 'number', 'driverRef', 'code', 'nationality', 'dob']],
                on="driverId", how="left"
            )

            # Verificando se 'number_y' foi criada
            if 'number_y' in results.columns:
                results.rename(columns={'number_y': 'number_drivers', 'nationality': 'nationality_driver'}, inplace=True)
            else:
                print("A coluna 'number_y' não foi encontrada após o merge.")

        # Relacionar com constructors
        results = results.merge(
            arquivos['constructors'][['constructorId', 'name', 'nationality']],
            on="constructorId", how="left"
        )
        results.rename(columns={'name': 'name_constructors', 'nationality': 'nationality_constructors'}, inplace=True)

        # Relacionar com status
        results = results.merge(
            arquivos['status'][['statusId', 'status']],
            on="statusId", how="left"
        )

        # Adicionar tempos de classificação (q1, q2, q3) da tabela 'qualifying'
        qualifying = arquivos['qualifying'].groupby('raceId').agg({
            'q1': 'first', 'q2': 'first', 'q3': 'first'
        }).reset_index()
        results = results.merge(qualifying, on="raceId", how="left")

        # Converter tempos para milissegundos nas colunas relevantes
        for col in ['fastestLapTime', 'q1', 'q2', 'q3']:
            if col in results.columns:
                results[col] = results[col].apply(convert_time_to_milliseconds)

        results['regulamento'] = results['year'].apply(
            lambda ano: next((nome for (inicio, fim), nome in {
                (2022, 2025): "2022-2025",
                (2017, 2021): "2017-2021",
                (2014, 2016): "2014-2016",
                (2009, 2013): "2009-2013",
                (2005, 2008): "2005-2008",
                (2001, 2004): "2001-2004",
                (1998, 2000): "1998-2000",
                (1995, 1997): "1995-1997",
                (1992, 1994): "1992-1994",
                (1989, 1991): "1989-1991",
                (1984, 1988): "1984-1988",
                (1981, 1983): "1981-1983",
                (1977, 1980): "1977-1980",
                (1973, 1976): "1973-1976",
                (1969, 1972): "1969-1972",
                (1966, 1968): "1966-1968",
                (1961, 1965): "1961-1965",
                (1958, 1960): "1958-1960",
                (1954, 1957): "1954-1957",
                (1951, 1953): "1951-1953",
                (1947, 1950): "1947-1950",
            }.items() if inicio <= ano <= fim), "Desconhecido")
        )

        # Substituir valores nulos em 'position' por "-1" na tabela correta (results)
        if 'position' in results.columns:
            results['position'] = results['position'].fillna(-1).astype(int).astype(str)
        else:
            logger.warning("A coluna 'position' não foi encontrada em 'results'.")

            
        # Preencher valores nulos com 0.0 nas colunas de tempo
        tempo_cols = ['fastestLapTime', 'fastestLapSpeed', 'q1', 'q2', 'q3']
        for col in tempo_cols:
            if col in results.columns:
                results[col] = results[col].fillna(0.0)
        
        colunas_finais = [
            'grid', 'position', 'points', 'year', 'name_circuits',
            'driverRef','dob', 'name_constructors', 
            'status', 'regulamento', 
            'fastestLapTime','fastestLapSpeed','q1', 'q2', 'q3'
        ]

        # colunas_finais = [
        #     'raceId', 'grid', 'position', 'points', 'year', 'name_circuits',
        #     'country', 'driverRef','number_drivers', 'code', 
        #     'nationality_driver', 'dob', 'name_constructors', 
        #     'nationality_constructors','status', 'regulamento', 
        #     'fastestLapTime','fastestLapSpeed', 'rank','q1', 'q2', 'q3'
        # ]

        results = results[colunas_finais]

        return results
    except Exception as e:
        logger.error(f"Erro ao ajustar os datasets: {e}")
        raise

def salvar_dataset(dataset, destino_final):
    try:
        os.makedirs(destino_final, exist_ok=True)
        dataset.to_csv(os.path.join(destino_final, "DatasetF1processado.csv"), index=False)
        logger.info("Dataset final salvo com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao salvar o dataset: {e}")
        raise

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    baixar_dataset(api)
