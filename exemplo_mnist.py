# =============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS (A "CAIXA DE FERRAMENTAS")
# =============================================================================
# Cada 'import' traz uma ferramenta essencial para o nosso trabalho.

import torch
# Biblioteca principal do PyTorch. Fornece a estrutura de dados central, o 'Tensor',
# e as operações que rodam em CPU ou GPU. É a base de tudo.

import torch.nn as nn
# O módulo 'nn' (Neural Networks) contém todos os blocos de construção para
# redes neurais, como camadas (Lineares, Convolucionais), funções de ativação, etc.
# O modelo ResNet que usaremos é construído com componentes deste módulo.

import torchvision.transforms as transforms
# 'torchvision' é a biblioteca do PyTorch para tarefas de Visão Computacional.
# O submódulo 'transforms' oferece um conjunto de ferramentas para pré-processar
# e transformar imagens (redimensionar, normalizar, etc.).

import torchvision.datasets as datasets
# Este submódulo do torchvision nos dá acesso fácil a datasets famosos,
# como o MNIST, permitindo baixá-los e carregá-los com uma única linha.

from torchvision.models import resnet18, ResNet18_Weights
# Do módulo de modelos pré-treinados, importamos duas coisas:
# 1. 'resnet18': A arquitetura específica do modelo de rede neural ResNet-18.
# 2. 'ResNet18_Weights': Um objeto que nos permite carregar os pesos pré-treinados
#    mais recentes e recomendados para a ResNet-18.

from torch.utils.data import DataLoader
# O DataLoader é uma ferramenta que pega um dataset e o organiza em pequenos lotes (batches),
# o que otimiza o uso de memória e acelera o processamento.



# =============================================================================
# CONFIGURAÇÃO INICIAL E PIPELINE DE TRANSFORMAÇÃO
# =============================================================================

# Define em qual dispositivo (CPU ou GPU) os cálculos serão realizados.
# Para este script, a CPU é suficiente.
device = torch.device("cpu")

# Define o tamanho do lote (batch size). Em vez de processar uma imagem por vez,
# processaremos 16 imagens simultaneamente para maior eficiência.
batch_size = 16

# 'transforms.Compose' cria um pipeline de pré-processamento. Cada imagem
# carregada passará por esta sequência de transformações.
transform = transforms.Compose([
    # 1. Redimensiona a imagem para 224x224 pixels. O ResNet-18 foi treinado
    #    com imagens deste tamanho, então precisamos que nossas imagens tenham o mesmo formato.
    #    Shape da imagem: [28, 28] -> [224, 224]
    transforms.Resize(224),

    # 2. O MNIST é em tons de cinza (1 canal), mas o ResNet-18 espera imagens coloridas (3 canais).
    #    Esta transformação duplica o canal de cinza três vezes para simular uma imagem RGB.
    #    Canais da imagem: 1 -> 3
    transforms.Grayscale(num_output_channels=3),

    # 3. Converte a imagem de um formato da biblioteca PIL para um Tensor do PyTorch.
    #    Também muda a ordem das dimensões para [Canais, Altura, Largura] e normaliza
    #    os valores dos pixels para o intervalo [0.0, 1.0].
    #    Shape: [224, 224, 3] -> [3, 224, 224]
    transforms.ToTensor(),

    # 4. Normaliza o tensor, ajustando os valores dos pixels para o intervalo [-1.0, 1.0].
    #    Isso ajuda a rede neural a funcionar de forma mais estável e eficiente.
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =============================================================================
# CARREGAMENTO DOS DADOS E DO MODELO
# =============================================================================

# Cria o objeto do dataset MNIST.
# root='./data': Onde salvar/procurar os dados.
# train=True: Usar o conjunto de treino (60.000 imagens).
# download=True: Baixar os dados se não forem encontrados localmente.
# transform=transform: Aplica o pipeline de transformações que definimos acima a cada imagem.
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)

# Cria o DataLoader. Ele pega o 'mnist_dataset' e o serve em lotes (batches).
# batch_size=batch_size: Agrupa os dados em lotes de 16.
# shuffle=False: Serve os dados na ordem original. Em um treinamento real,
#                usaríamos shuffle=True para embaralhar os dados a cada época.
mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)


# Carrega a arquitetura ResNet-18 e preenche ela com os pesos pré-treinados
# no dataset ImageNet. Estes pesos contêm o "conhecimento" acumulado da rede.
# .to(device): Move o modelo para o dispositivo definido (CPU).
resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

# ATENÇÃO: O comando .eval() é crucial. Ele coloca o modelo em "modo de avaliação".
# Isso desativa camadas como o 'Dropout' e "congela" o comportamento do 'Batch Normalization',
# garantindo que o resultado para uma mesma imagem seja sempre o mesmo (determinístico).
# Se estivéssemos treinando, usaríamos o modo .train().
resnet.eval()

# =============================================================================
# O "PULO DO GATO": EXTRAÇÃO DE FEATURES COM UM FORWARD HOOK
# =============================================================================
# O objetivo não é classificar, mas sim extrair o vetor de características (features)
# de uma camada intermediária. Para fazer isso sem alterar o código do ResNet,
# usamos um "hook", que funciona como um "espião" em uma camada.

# Lista vazia para armazenar as features capturadas a cada lote.
extracted_features = []

# Esta é a função "espiã" (o hook). Ela será chamada automaticamente toda vez que
# dados passarem pela camada que estamos monitorando.
def hook_fn(module, input, output):
    # 'output' é o tensor de saída da camada. Nós o desvinculamos do grafo de
    # computação (.detach()), movemos para a CPU (.cpu()) e o adicionamos à nossa lista.
    extracted_features.append(output.detach().cpu())

# Registrando o hook na camada 'avgpool' da ResNet.
# A camada 'avgpool' é a última camada antes da classificação final, e sua saída
# é um vetor de características compacto e de alto nível (512 dimensões na ResNet-18).
# A variável 'hook_handle' nos permite remover o hook mais tarde.
hook_handle = resnet.avgpool.register_forward_hook(hook_fn)

# =============================================================================
# PROCESSAMENTO DOS DADOS (O LOOP PRINCIPAL)
# =============================================================================
# Agora, vamos passar todas as imagens pela rede para que o hook possa capturá-las.

# Lista para guardar os rótulos (labels) de cada imagem.
all_labels = []

# 'with torch.no_grad():' é um comando de otimização. Ele desativa o cálculo
# de gradientes, o que economiza memória e acelera muito o processo, já que
# não estamos treinando a rede, apenas usando-a para inferência.
with torch.no_grad():
    # 'for images, labels in mnist_loader:' é uma sintaxe Python chamada "desempacotamento".
    # A cada iteração, o 'mnist_loader' fornece um lote que é uma lista de dois elementos:
    # [tensor_de_imagens, tensor_de_rotulos]. O Python automaticamente atribui o primeiro
    # à variável 'images' e o segundo à variável 'labels'.
    for images, labels in mnist_loader:
        # Move o lote de imagens para o dispositivo (CPU).
        images = images.to(device)

        # Esta é a passagem dos dados pela rede (forward pass).
        # O resultado final da classificação (o que a ResNet acha que o dígito é)
        # não nos interessa, por isso o atribuímos a '_' (uma convenção para "ignorar").
        # A MÁGICA ACONTECE AQUI: ao executar esta linha, os dados passam pela camada 'avgpool',
        # o nosso hook é ativado, e as features são salvas na lista 'extracted_features'.
        _ = resnet(images)

        # Guarda os rótulos do lote atual.
        all_labels.append(labels)

# =============================================================================
# ORGANIZAÇÃO FINAL DOS DADOS
# =============================================================================
# Após o loop, temos duas listas ('extracted_features' e 'all_labels') cheias de
# pequenos tensores (um para cada lote). Agora, vamos juntá-los.

# 'torch.cat' concatena todos os tensores da lista em um único tensor gigante.
# A lista 'extracted_features' tinha 3750 tensores de shape [16, 512, 1, 1].
# Após o 'cat', teremos um único tensor de shape [60000, 512, 1, 1].
features = torch.cat(extracted_features, dim=0)

# O método '.view()' remodela o tensor. O '-1' é um coringa que diz ao PyTorch
# para calcular a dimensão automaticamente.
# Isso "achata" o tensor, removendo as dimensões extras de tamanho 1.
# Shape: [60000, 512, 1, 1] -> [60000, 512]
features = features.view(features.size(0), -1)

# Fazemos o mesmo para os rótulos, juntando os 3750 tensores de shape [16]
# em um único tensor de shape [60000].
labels = torch.cat(all_labels)

# Imprime o shape (formato) final dos tensores para verificar se tudo correu bem.
print("Features extraídas:", features.shape) # Deve ser [60000, 512]
print("Labels:", labels.shape)           # Deve ser [60000]

# É uma boa prática remover o hook quando não precisamos mais dele para
# evitar que ele continue consumindo recursos em usos futuros do modelo.
hook_handle.remove()