from kaggle.api.kaggle_api_extended import KaggleApi

def autenticar_kaggle():
    """
    Autentica no Kaggle usando a API.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Erro ao autenticar no Kaggle: {e}")
        return None