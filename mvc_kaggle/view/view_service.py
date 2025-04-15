import shap
import os
import matplotlib.pyplot as plt

def display_results(shap_summary, output_dir="mvc_kaggle/view/relatorios"):
    """
    Exibe os resultados do treinamento e salva a imagem do gráfico SHAP.
    """
    print("Model training and feature analysis completed.")
    
    # Garantir que o diretório de saída exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Caminho do arquivo para salvar o gráfico SHAP
    shap_file = os.path.join(output_dir, "shap_F1Dataset.png")
    
    # Gerar o gráfico SHAP e salvar como imagem
    print("Gerando gráfico SHAP...")
    shap.summary_plot(shap_summary, show=False)  # Gera o gráfico sem exibir
    plt.savefig(shap_file)  # Salva o gráfico no arquivo especificado
    plt.close()  # Fecha o gráfico para evitar sobreposição em futuras execuções
    print(f"Gráfico SHAP salvo em: {shap_file}")