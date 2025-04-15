import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV."""
    data = pd.read_csv(file_path)
    data = data.sample(frac=0.001, random_state=1179)
    return data