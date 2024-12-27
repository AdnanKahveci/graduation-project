import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Modeli yükleme
model = torch.load('checkpoint/best_model.pt')  # Modeli yükleyin (modelin tam yolunu belirtin)

# Modelin yapısını inceleyin
#print(model)

# Ağırlıkları çıkartma (Modelin doğru yoluna göre değişebilir)
weights = model['module.blocks.0.flows.14.coupling.net.2.weight']

actnorm_scale = model["module.blocks.0.flows.14.actnorm.scale"]
actnorm_loc = model["module.blocks.0.flows.14.actnorm.loc"]
coupling_weight = model["module.blocks.0.flows.14.coupling.net.2.weight"]
conv_weight = model["module.blocks.0.flows.14.coupling.net.4.conv.weight"]

# Ağırlıkları 2D'ye indirgeme ve CPU'ya taşıma
weights_2d = weights.squeeze().detach().cpu().numpy()

# Isı haritası ile görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(weights_2d, cmap='coolwarm')
plt.title('Ağırlık Isı Haritası')
plt.show()

# Ağırlıkları düzleştirip histogram çizme
weights_2d_flattened = weights_2d.flatten()

plt.figure(figsize=(10, 6))
plt.hist(weights_2d_flattened, bins=50, color='b', edgecolor='k')
plt.title('Ağırlık Dağılımı (Histogram)')
plt.xlabel('Ağırlık Değeri')
plt.ylabel('Frekans')
plt.show()

# 3D yüzey grafiği
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# X ve Y eksenlerini oluşturma
X = np.arange(weights_2d.shape[0])
Y = np.arange(weights_2d.shape[1])
X, Y = np.meshgrid(X, Y)

# Yüzeyi çizme
ax.plot_surface(X, Y, weights_2d, cmap='viridis')

plt.title('Ağırlık 3D Yüzey Grafiği')
plt.show()
