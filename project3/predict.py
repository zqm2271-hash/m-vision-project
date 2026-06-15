import numpy as np
import pickle
import random
import cv2
import matplotlib.pyplot as plt
import os

from extract_features import extract_bovw_feature

idf = np.load("idf.npy")
clf = pickle.load(open("svm_model.pkl", "rb"))

test_paths = np.load("test_paths.npy", allow_pickle=True)
test_labels = np.load("test_labels.npy")

classes = sorted(os.listdir(r"D:\lyxxx\3\data\15-Scene"))

# =========================
# 随机抽10张
# =========================
idxs = random.sample(range(len(test_paths)), 10)

plt.figure(figsize=(15, 8))

for i, idx in enumerate(idxs):

    img_path = test_paths[idx]
    true_label = test_labels[idx]

    # 提取特征
    feature = extract_bovw_feature(img_path)
    feature = feature * idf

    # 预测
    pred = clf.predict([feature])[0]

    # 读图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 判断对错
    result = "✔" if pred == true_label else "✘"

    # 显示
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")

    plt.title(
        f"T:{classes[true_label]}\nP:{classes[pred]}\n{result}",
        fontsize=9
    )

plt.tight_layout()
plt.show()