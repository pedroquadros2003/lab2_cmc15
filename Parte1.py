## Este módulo é para trabalhar os datasets, neste caso de imagens, com a unidade fundamental dos tensores
import torch

## Esse submódulo de torch é para criar o dataset a partir de uma pasta no computador e depois carregá-lo
from torch.utils.data import Dataset, DataLoader

## Estes módulo é para realizar transformações sobre as imagens
import torchvision.transforms as transforms
from PIL import Image

## Este módulo permite a leitura de arquivos no python e operações com caminhos
import os

## Estes módulos são as redes neurais, já treinadas, que utilizaremos para gerar características para as imagens do dataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


## Este objeto transform é uma pipeline de ações para padronizar o tipo de imagem do nosso dataset, retornando tensores do torch ao final
transform = transforms.Compose([
    transforms.Resize(256),         
    transforms.CenterCrop(224),  
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



## Classe criada para realizar criar o dataset a partir dos caminhos fornecidos na máquina onde este código é rodado

class TxtFileDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths_and_labels = []

        with open(txt_file, 'r') as f:
            for line in f:
                # Divide cada linha em uma lista com o caminho e com o rótulo, respectivamente
                line = line.strip()
                if line:
                    path, label = line.split()
                    self.image_paths_and_labels.append((path, int(label)))

    def __len__(self):
        # Retorna o número total de amostras no dataset
        return len(self.image_paths_and_labels)

    def __getitem__(self, idx):
        ## Será útil para situações em que precisaremos fazer dataset[idx]

        relative_path, label = self.image_paths_and_labels[idx]
        
        full_image_path = os.path.join(self.root_dir, relative_path)
        
        # Carrega a imagem usando a biblioteca Pillow (PIL)
        # .convert('RGB') garante que a imagem tenha 3 canais
        image = Image.open(full_image_path).convert('RGB')

        # Aplica as transformações na imagem, se houver
        if self.transform:
            image = self.transform(image)

        return image, label




caminho_arquivo_txt = f'.{os.sep}Dados{os.sep}train.txt'        # caminho para train.txt
diretorio_raiz_dados = f'.{os.sep}Dados'                        # caminho para o diretório de Dados


## Criação do dataset
dataset_train = TxtFileDataset(txt_file=caminho_arquivo_txt,
                             root_dir=diretorio_raiz_dados,
                             transform=transform)

## Estabelece-se que será utilizada a CPU para os processos com as redes neurais
device = torch.device("cpu")

## Para obtermos um processamento mais rápido, rodaremos as redes neurais em lotes de 16 imagens
batch_size = 16

data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

################################################################### 

## Aqui, definimos a rede neural que será utilizada

#resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

################################################################### 


## Estabelecemos que não iremos treinar a rede, mas testá-la, avaliá-la, tomando o seu tensor intermediário logo antes da classificação final
resnet.eval()
extracted_features = []

## Definimos a função hook para extrair o tensor intermediário do processo das redes neurais

def hook_fn(module, input, output):
    extracted_features.append(output.detach().cpu())


# Registrando o hook na camada 'avgpool'(camada antes da camada de classificação)
hook_handle = resnet.avgpool.register_forward_hook(hook_fn)
all_labels = []


## Aplicação da rede neural sobre as imagens no dataset já carregado
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        _ = resnet(images)
        all_labels.append(labels)

## Ajuste da dimensão dos tensores que guardam o resultado intermediário das redes neurais quando alimentadas com o dataset
features = torch.cat(extracted_features, dim=0)
features = features.view(features.size(0), -1)
labels = torch.cat(all_labels)

## Salvando os resultados obtidos para acessá-los diretamente depois
torch.save({
    'features': features,
    'labels': labels
}, 'lixo.pt')


print("Features extraídas:", features.shape) # Deve ser [170, 512] ou [170, 2048]
print("Labels:", labels.shape) # [170]
hook_handle.remove()
