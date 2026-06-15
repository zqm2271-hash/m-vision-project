import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from extract_features import extract_bovw_feature

import os

DATASET_PATH = r"D:\lyxxx\3\data\15-Scene"

classes = sorted(os.listdir(DATASET_PATH))
idf = np.load("idf.npy")

# =========================
# 1. 每个类别取“平均特征”
# =========================
class_features = []

for cls in classes:

    folder = os.path.join(DATASET_PATH, cls)

    feats = []

    for img in os.listdir(folder):

        if img.endswith(".jpg"):

            path = os.path.join(folder, img)

            f = extract_bovw_feature(path)
            f = f * idf

            feats.append(f)

    feats = np.array(feats)

    mean_feat = np.mean(feats, axis=0)

    class_features.append(mean_feat)

class_features = np.array(class_features)

# =========================
# 2. 欧氏距离矩阵
# =========================
dist_matrix = pairwise_distances(
    class_features,
    metric='euclidean'
)

# =========================
# 3. 可视化
# =========================
plt.figure(figsize=(8, 6))

plt.imshow(dist_matrix, cmap='hot')

plt.colorbar()

plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

plt.title("Class-wise Euclidean Distance Matrix")

plt.tight_layout()
plt.show()