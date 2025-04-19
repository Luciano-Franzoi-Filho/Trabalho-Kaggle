from pycaret.regression import *
import os
import matplotlib.pyplot as plt

def train_with_pycaret(data, target):
    """
    Treina modelos usando o módulo de regressão do PyCaret e gera um gráfico com os resultados.
    
    Args:
        data (pd.DataFrame): O dataset contendo as features e o target.
        target (str): O nome da coluna alvo.
    
    Returns:
        best_model: O melhor modelo treinado pelo PyCaret.
    """
    # Configura o ambiente do PyCaret
    exp = setup(data=data, target=target, verbose=False, session_id=123)
    
    # Compara os modelos e seleciona o melhor
    best_model = compare_models()
    
    # Salvar o melhor modelo na pasta especificada
    output_dir = "mvc_kaggle/view/relatorios"
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório exista
    model_path = os.path.join(output_dir, 'best_model')
    save_model(best_model, model_path)
    print(f"Melhor modelo salvo em: {model_path}.pkl")
    
    # Gerar relatório de todos os treinamentos realizados
    results = pull()  # Obtém os resultados de todos os modelos treinados
    print("Resultados dos modelos treinados:")
    print(results)  # Verificar o conteúdo do DataFrame
    
    # Escolher uma métrica válida para o gráfico
    metric = 'MAE' if 'MAE' in results.columns else 'RMSE'
    
    # Criar gráfico com os resultados
    plt.figure(figsize=(10, 6))
    plt.bar(results['Model'], results[metric], color='skyblue')  # Usar a métrica disponível
    plt.xlabel('Modelos')
    plt.ylabel(metric)
    plt.title(f'Comparação de Modelos - PyCaret ({metric})')
    plt.xticks(rotation=45, ha='right')
    
    # Salvar o gráfico na pasta especificada
    graph_path = os.path.join(output_dir, 'pycaret_training_graph.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()
    print(f"Gráfico de treinamentos salvo em: {graph_path}")
    
    return best_model