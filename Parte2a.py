import torch
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Carrega o dicionário salvo previamente salvo em arquivo
dados_carregados = torch.load('resnet50_data.pt')

# Acessa os tensores usando as mesmas chaves
features = dados_carregados['features']
labels = dados_carregados['labels']

# Feature extraction
test = SelectKBest(score_func=chi2,k=5)
fit = test.fit(features, labels)
features = fit.transform(features)
labels = labels.numpy()



## Salvando os resultados obtidos, já no formato de array numpy, para acessá-los diretamente depois
torch.save({
    'features': features,
    'labels': labels
}, 'select_data_np.pt')


print("Features extraídas:", features.shape) # Deve ser [170, 5]
print("Labels:", labels.shape) # [170]
print(type(labels))
print(type(features))
