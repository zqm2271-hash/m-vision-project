import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# ==========================
# 设备
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

DATASET_PATH = r"D:\lyxxx\4\data\15-Scene"

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes

# ==========================
# 加载模型
# ==========================
model = SimpleCNN(len(class_names)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ==========================
# 预测函数
# ==========================
def show_predictions(num=10):

    idxs = random.sample(range(len(dataset)), num)

    plt.figure(figsize=(15,6))

    for i, idx in enumerate(idxs):

        img, label = dataset[idx]

        input_img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            pred = torch.argmax(output, dim=1).item()

        plt.subplot(2,5,i+1)
        plt.imshow(img.permute(1,2,0))
        plt.title(f"GT:{class_names[label]}\nPred:{class_names[pred]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

show_predictions(10)