import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega o dicionário salvo previamente salvo em arquivo
dados_carregados = torch.load('select_data_np.pt', weights_only=False)

# Carrega os arrays numpy
features = dados_carregados['features']
labels = dados_carregados['labels']

def criar_box_plot(idx):
    feature_a_plotar = features[:, idx]

    # Crie um DataFrame do Pandas. Esta é a estrutura de dados ideal para o Seaborn.
    # Teremos duas colunas: uma para os valores da feature e outra para os rótulos.
    df = pd.DataFrame({
        'feature_value': feature_a_plotar,
        'label': labels
    })


    plt.figure(figsize=(10, 6))

    sns.boxplot(x='label', y='feature_value', data=df)

    plt.title(f'Boxplot da Feature {idx} para cada Classe')
    plt.xlabel('Classe (Label)')
    plt.ylabel(f'Valor da Feature {idx}')
    plt.grid(True) # Adiciona uma grade para facilitar a leitura
    plt.show()



for idx in range(0,5):
    criar_box_plot(idx)