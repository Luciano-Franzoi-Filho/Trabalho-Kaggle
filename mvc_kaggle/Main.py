import os
import traceback
import pandas as pd
from controller.kaggle_authenticator import autenticar_kaggle
from controller.dataset_downloader import baixar_dataset
from model.eda_service import gerar_relatorio
from controller.pipeline_controller import run_pipeline
from view.relatorios_geral import gerar_graficos_relatorios

def main():
    """
    Ponto de entrada do projeto. Executa autenticação, download do dataset, EDA,
      geração de relatórios, treinamento e seleção de variáveis.
    """
    print("Iniciando o sistema...")

    try:
        # Etapa 1: Autenticação no Kaggle
        print("Autenticando no Kaggle...")
        api = autenticar_kaggle()
        if not api:
            print("Erro na autenticação. Encerrando o sistema.")
            return
        print("Autenticação concluída com sucesso!")

        # Etapa 2: Download do Dataset
        print("Baixando o dataset...")
        baixar_dataset(api)
        print("Download do dataset concluído!")

        # Etapa 3: Análise Exploratória de Dados (EDA)
        print("Gerando relatório de EDA...")
        gerar_relatorio()
        print("Relatório de EDA gerado com sucesso!")

        # Caminho do dataset processado
        dataset_path = "mvc_kaggle/model/Data/Processed/DatasetF1processado.csv"
        if not os.path.exists(dataset_path):
            print(f"Arquivo {dataset_path} não encontrado. Encerrando o sistema.")
            return

        # Etapa 4: Geração de Relatórios (Gráficos)
        print("Gerando gráficos e relatórios...")
        gerar_graficos_relatorios()  # Gera e salva os gráficos
        print("Relatórios gerados com sucesso!")

        # Etapa 5: Treinamento do Modelo
        print("Iniciando o pipeline...")
        run_pipeline(dataset_path, 1)  # mínimo possível é 0.3 
        print("Pipeline concluído com sucesso!")

    # except Exception as e:
    #     print(f"Erro inesperado durante a execução: {e}")

    except Exception as e:
        print("Ocorreu um erro:")
        traceback.print_exc()  # Exibe o rastreamento completo do erro

if __name__ == "__main__":
    main()