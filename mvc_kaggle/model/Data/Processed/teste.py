import pandas as pd

# Caminho para o arquivo CSV
caminho_arquivo = 'mvc_kaggle\model\Data\Processed\DatasetF1processado.csv'  # Substitua pelo nome real do seu arquivo

# Lê o CSV com o pandas
df = pd.read_csv(caminho_arquivo)

# Mostra as colunas disponíveis
print("Colunas disponíveis:", list(df.columns))

# Solicita o nome da coluna e o valor a ser buscado
coluna = input("Digite o nome da coluna onde deseja buscar: ")
valor = input("Digite o valor que deseja buscar: ")

# Filtra o DataFrame para encontrar o valor
resultado = df[df[coluna].astype(str) == valor]

# Exibe o resultado
if not resultado.empty:
    print("Valor encontrado nas seguintes linhas:")
    print(resultado)
else:
    print("Valor não encontrado.")