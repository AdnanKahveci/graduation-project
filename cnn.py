import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HAM10000 ve üretilmiş veri yolu
ham_data_path = "ham10000/HAM10000_images_part_1"
generated_data_path = "sample"

# Veriyi dönüştürme işlemleri
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Görüntü boyutlarını Glow modeline uyacak şekilde ayarlayın
    transforms.ToTensor()
])

# Özel veri kümesi sınıfı, görüntüleri yükler ve dönüşümlerini yapar
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, label, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.label = label
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label

# Orijinal HAM10000 veri kümesi (label 0)
ham_dataset = CustomImageDataset(ham_data_path, label=0, transform=transform)

# Üretilen görüntüler veri kümesi (label 1)
generated_dataset = CustomImageDataset(generated_data_path, label=1, transform=transform)

# Veri kümelerini birleştirme
combined_dataset = ConcatDataset([ham_dataset, generated_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
import torch.nn as nn
import torch.optim as optim

# Basit CNN modeli tanımlama
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli, kaybı ve optimizasyonu tanımlama
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in combined_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # İleri + geri + optimizasyon
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(combined_loader)}")

print("Eğitim tamamlandı.")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Tahminleri ve etiketleri toplama
true_labels = []
pred_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in combined_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Doğruluk ve karışıklık matrisi
accuracy = accuracy_score(true_labels, pred_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels)

print(f"Doğruluk: {accuracy*100:.2f}%")

# Karışıklık matrisini görselleştirme
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['HAM10000', 'Generated'], yticklabels=['HAM10000', 'Generated'])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Karışıklık Matrisi")
plt.show()
