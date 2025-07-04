import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_pairplot(csv_file='DBSCAN_result.csv'):
    """
    Carrega os dados de clusterização e cria um pairplot para visualizar
    todas as combinações de características, coloridas pelo cluster.
    """
    # Carrega os dados do arquivo CSV
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{csv_file}' não foi encontrado.")
        print("Certifique-se de que o arquivo está no mesmo diretório que o script.")
        return

    print("Dados carregados. Gerando o gráfico de pares (pairplot)...")

    # Define um estilo estético para o gráfico
    sns.set_theme(style="ticks")

    # Cria o pairplot
    # A função irá automaticamente usar todas as colunas numéricas do DataFrame.
    # A coluna 'TrueClass' (texto) será ignorada nos eixos, o que é o desejado.
    pair_plot = sns.pairplot(
        df,
        hue='Cluster',       # Colore os pontos pela coluna 'Cluster'
        palette='pastel',    # Usa a paleta de cores "pastel"
        diag_kind='kde',     # Mostra a distribuição de densidade na diagonal
        plot_kws={'alpha': 0.8, 's': 50},  # Ajusta a transparência e tamanho dos pontos
        diag_kws={'fill': True} # Preenche a área dos gráficos de densidade
    )
    
    # Adiciona um título geral acima do gráfico
    pair_plot.fig.suptitle("Matriz de Dispersão (Pairplot) por Cluster DBSCAN", y=1.02, fontsize=16)

    # Salva a figura em um arquivo
    plt.savefig("DBSCAN_pairplot.png")

    # Mostra o gráfico
    plt.show()


# Executa a função
if __name__ == '__main__':
    #create_pairplot('DBSCAN_result.csv')
    create_pairplot('Kohonen_result.csv')
