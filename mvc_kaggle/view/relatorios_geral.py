import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Garante que o diretório de saída exista
output_dir = 'mvc_kaggle/view/relatorios'
os.makedirs(output_dir, exist_ok=True)

def gerar_graficos_relatorios():
    # Caminho do arquivo CSV
    csv_path = 'mvc_kaggle/model/Data/Processed/DatasetF1processado.csv'

    # Carrega o dataset corrigido
    df = pd.read_csv(csv_path)

    # Ajusta os tipos
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Adiciona idade
    df['age'] = df['year'] - df['dob'].dt.year

    # 1. Evolução de pontos por equipe ao longo dos anos (2010+)
    team_year = df[df['year'] >= 2010].groupby(['year', 'name_constructors'])['points'].sum().reset_index()

    # Seleciona as top 8 equipes com mais pontos desde 2010
    top_equipes_2010 = team_year.groupby('name_constructors')['points'].sum().nlargest(8).index.tolist()
    df_top_equipes_2010 = team_year[team_year['name_constructors'].isin(top_equipes_2010)]

    # Definindo as cores personalizadas das equipes
    equipes_cores = {
        'Mercedes': '#38B09D',  # Petronas Green
        'Force India': '#FF6600',  # Caramelo
        'Renault': '#FFCD00',  # Amarelo
        'Ferrari': '#DC0000',  # Vermelho
        'Red Bull': '#1E41FF',  # Azul escuro
        'McLaren': '#FF8700',  # Laranja
        'Lotus F1': '#006F62',  # Verde escuro
        'Williams': '#005A8D'  # Azul claro
    }

    # Plotando gráfico de linha com cores personalizadas
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df_top_equipes_2010,
        x='year',
        y='points',
        hue='name_constructors',
        marker='o',
        linewidth=4,
        palette=equipes_cores  # Usando as cores definidas
    )

    plt.title('Evolução de Pontos por Temporada (2010+) - Top 8 Equipes', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Pontos por Temporada', fontsize=12)
    plt.xticks(df_top_equipes_2010['year'].unique())  # Marcação em cada ano
    plt.grid(True, linestyle='--', alpha=0.3)

    # Legenda ajustada
    plt.legend(title='Equipes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evolucao_pontos_equipes_2010.png', bbox_inches='tight')
    plt.close()


    # 2. Pontos totais por piloto
    plt.figure(figsize=(12, 6))
    piloto_pontos = df.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(20)
    sns.barplot(x=piloto_pontos.values, y=piloto_pontos.index)
    plt.title('Top 20 Pilotos com Mais Pontos Acumulados')
    plt.xlabel('Pontos Totais')
    plt.ylabel('Piloto')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pontos_por_piloto.png')
    plt.close()

    # 3. Idade vs posição (somente posições válidas)
    df_validos = df[df['position'] > 0]

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Pontos de cada corrida
    scatter = sns.scatterplot(
        data=df_validos,
        x='age',
        y='position',
        alpha=0.4,
        hue='year',
        palette='viridis',
        edgecolor=None,
        legend=False
    )

    # Linha de tendência (regressão linear)
    sns.regplot(
        data=df_validos,
        x='age',
        y='position',
        scatter=False,
        color='crimson',
        line_kws={"linewidth": 4, "linestyle": '-'},
        ci=None
    )

    # Títulos e eixos
    plt.title('🔍 Relação entre Idade do Piloto e sua Posição Final na Corrida', fontsize=14)
    plt.xlabel('Idade do Piloto (em anos)', fontsize=12)
    plt.ylabel('Posição Final na Corrida (1º é melhor)', fontsize=12)

    # Inverter o eixo Y para que 1 fique no topo (posição melhor)
    plt.gca().invert_yaxis()

    # Grid mais suave
    plt.grid(True, linestyle='--', alpha=0.3)

    # Salvar figura
    plt.tight_layout()
    plt.savefig(f'{output_dir}/idade_vs_posicao.png')
    plt.close()

    # 4. Melhores equipes de todos os tempos
    melhores_equipes = df.groupby('name_constructors')['points'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=melhores_equipes.values, y=melhores_equipes.index)
    plt.title('Top 10 Equipes com Mais Pontos Acumulados')
    plt.xlabel('Pontos Totais')
    plt.ylabel('Equipe')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/melhores_equipes.png')
    plt.close()

    # 5. Melhores pilotos de todos os tempos
    melhores_pilotos = piloto_pontos.head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=melhores_pilotos.values, y=melhores_pilotos.index, palette='mako')
    plt.title('Top 10 Pilotos com Mais Pontos')
    plt.xlabel('Pontos Totais')
    plt.ylabel('Piloto')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/melhores_pilotos.png')
    plt.close()

    # 6. Status mais comuns em abandonos
    dnf = df[df['position'] == -1]
    status_counts = dnf['status'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=status_counts.values, y=status_counts.index, palette='rocket')
    plt.title('Status Mais Comuns em Abandonos')
    plt.xlabel('Quantidade de Ocorrências')
    plt.ylabel('Status')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/status_abandonos.png')
    plt.close()
