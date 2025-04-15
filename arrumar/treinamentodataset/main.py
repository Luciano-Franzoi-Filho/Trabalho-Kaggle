import traceback
from model_pipeline import run_pipeline, predict_2025

def main(file_path: str):
    """
    Função principal para executar o pipeline de treinamento de modelos e fazer previsões para 2025.
    """
    try:
        # Executar o pipeline para treinar o modelo das top 5 equipes
        print("Iniciando o pipeline de treinamento...")
        modelo_equipes, preprocessor_equipes = run_pipeline(file_path)
        print("Pipeline de treinamento concluído com sucesso!")

        # Fazer previsões para a temporada de 2025
        print("Fazendo previsões para 2025...")
        predict_2025(modelo_equipes, preprocessor_equipes)
        print("Previsões para 2025 concluídas!")

    except Exception as e:
        print("Ocorreu um erro:")
        traceback.print_exc()  # Exibe o rastreamento completo do erro

if __name__ == '__main__':
    file_path = "mvc_kaggle/model/dataset_arrumado/corridas_geral.csv"
    
    # Chamar a função main
    main(file_path)
    