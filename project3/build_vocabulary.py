import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

DATASET_PATH = r"D:\lyxxx\3\data\15-Scene"
K = 200

sift = cv2.SIFT_create()

all_paths = []
all_labels = []

print("读取数据...")

classes = sorted(os.listdir(DATASET_PATH))

for label, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)

    for img in os.listdir(folder):
        if img.lower().endswith(".jpg"):
            all_paths.append(os.path.join(folder, img))
            all_labels.append(label)

# 划分 train/test（关键）
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print("提取train SIFT...")

all_des = []

for p in train_paths:
    img = cv2.imread(p, 0)
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        all_des.append(des)

all_des = np.vstack(all_des)

print("KMeans训练...")

kmeans = MiniBatchKMeans(
    n_clusters=K,
    batch_size=1000,
    random_state=42
)

kmeans.fit(all_des)

np.save("vocabulary.npy", kmeans.cluster_centers_)

# 保存划分
np.save("train_paths.npy", train_paths)
np.save("test_paths.npy", test_paths)
np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)

print("完成 vocabulary + train/test split")