import os
import pandas as pd
import sweetviz as sv
import dtale
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mvc_kaggle/logs/logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EDA_Project")

def carregar_dados(diretorio="mvc_kaggle/model/Data/Processed"):
    """
    Carrega todos os arquivos CSV de um diretório especificado em um dicionário de DataFrames.
    """
    try:
        if not os.path.exists(diretorio):
            raise FileNotFoundError(f"O diretório '{diretorio}' não foi encontrado.")
        
        arquivos_csv = [arq for arq in os.listdir(diretorio) if arq.endswith('.csv')]
        if not arquivos_csv:
            raise FileNotFoundError(f"Nenhum arquivo CSV foi encontrado no diretório '{diretorio}'.")
        
        dados = {}
        for arquivo in arquivos_csv:
            caminho_arquivo = os.path.join(diretorio, arquivo)
            nome_arquivo = os.path.splitext(arquivo)[0]
            dados[nome_arquivo] = pd.read_csv(caminho_arquivo)
        
        return dados
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        return {}

class EDAReport:
    def __init__(self, nome_arquivo: str, df: pd.DataFrame, output_dir: str):
        self.nome_arquivo = nome_arquivo
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_autoviz(self):
        """ Gera relatório AutoViz """
        try:
            av = AutoViz_Class()
            av.AutoViz(filename="", dfte=self.df)
            logger.info(f"Relatório AutoViz gerado para {self.nome_arquivo}.")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório AutoViz para {self.nome_arquivo}: {e}")

    def generate_dtale(self):
        """ Inicia a interface interativa D-Tale """
        try:
            dtale.show(self.df)
            logger.info(f"D-Tale iniciado para {self.nome_arquivo}.")
        except Exception as e:
            logger.error(f"Erro ao iniciar D-Tale para {self.nome_arquivo}: {e}")

    # def generate_sweetviz(self):
    #     """ Gera relatório Sweetviz """
    #     try:
    #         report = sv.analyze(self.df)
    #         output_path = os.path.join(self.output_dir, f"sweetviz_{self.nome_arquivo}.html")
    #         report.show_html(output_path)
    #         logger.info(f"Relatório Sweetviz salvo em {output_path}.")
    #     except Exception as e:
    #         logger.error(f"Erro ao gerar relatório Sweetviz para {self.nome_arquivo}: {e}")

    def generate_ydata(self):
        """ Gera relatório YData Profiling """
        try:
            profile = ProfileReport(self.df.sample(frac=0.1, random_state=51), explorative=True)
            output_path = os.path.join(self.output_dir, f"ydata_{self.nome_arquivo}.html")
            profile.to_file(output_path)
            logger.info(f"Relatório YData Profiling salvo em {output_path}.")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório YData para {self.nome_arquivo}: {e}")

def gerar_relatorio():
    """ Executa as análises EDA para os datasets juntos e segmentados """
    diretorio = "mvc_kaggle/model/Data/Processed"
    output_dir = "mvc_kaggle/view/relatorios"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Iniciando análises para o diretório: {diretorio}...")
    dados = carregar_dados(diretorio)
    if not dados:
        logger.warning(f"Nenhum dado encontrado no diretório: {diretorio}.")
        return
    
    for nome_arquivo, df in dados.items():
        if df.empty:
            logger.warning(f"O arquivo {nome_arquivo} está vazio, pulando...")
            continue
        
        logger.info(f"Processando {nome_arquivo} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        eda = EDAReport(nome_arquivo, df, output_dir)
        eda.generate_autoviz()
        # eda.generate_sweetviz()
        eda.generate_ydata()
        eda.generate_dtale()
    
    logger.info("Todas as análises foram concluídas com sucesso!")

def exibir_informacoes_base():
    """
    Exibe informações detalhadas da base de dados da Fórmula 1.
    """
    try:
        dados = carregar_dados()
        if not dados:
            logger.warning("Nenhum dado foi carregado. Verifique o diretório do dataset.")
            return
        
        logger.info("--- Informações da Base de Dados de Fórmula 1 ---")
        for nome_arquivo, df in dados.items():
            logger.info(f"Arquivo: {nome_arquivo}")
            logger.info(f"Resumo estatístico:\n{df.describe()}")
            buffer = []
            df.info(buf=buffer)
            logger.info("\n".join(buffer))
        logger.info("Informações exibidas com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao exibir informações da base: {e}")

if __name__ == "__main__":
    # Executar as análises EDA
    gerar_relatorio()

    # Exibir informações da base de dados
    exibir_informacoes_base()