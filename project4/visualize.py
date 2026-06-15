import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

# ==========================
# 设备
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# CNN
# ==========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

# ==========================
# 数据
# ==========================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
DATASET_PATH = r"D:\lyxxx\4\data\15-Scene"

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

class_names = dataset.classes

# ==========================
# 模型
# ==========================
model = SimpleCNN(len(class_names)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ==========================
# 混淆矩阵
# ==========================
y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in loader:

        imgs = imgs.to(device)

        outputs = model(imgs)

        preds = torch.argmax(outputs,1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==========================
# PCA可视化
# ==========================
features = []
labels_list = []

with torch.no_grad():
    for imgs, labels in loader:

        imgs = imgs.to(device)

        feat = model.features(imgs)
        feat = feat.view(feat.size(0), -1)

        features.append(feat.cpu().numpy())
        labels_list.extend(labels.numpy())

features = np.concatenate(features, axis=0)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(8,6))

for i in range(len(class_names)):
    idx = np.array(labels_list) == i
    plt.scatter(reduced[idx,0], reduced[idx,1], s=10, label=class_names[i])

plt.legend()
plt.title("PCA Feature Visualization")
plt.show()