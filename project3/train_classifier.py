import numpy as np
import pickle
from sklearn.svm import SVC
from extract_features import extract_bovw_feature

train_paths = np.load("train_paths.npy", allow_pickle=True)
train_labels = np.load("train_labels.npy")

print("提取训练特征...")

X = []

for p in train_paths:
    X.append(extract_bovw_feature(p))

X = np.array(X)

# =========================
# TF-IDF（关键提升）
# =========================
df = np.sum(X > 0, axis=0)
idf = np.log((len(X) + 1) / (df + 1)) + 1

X = X * idf

np.save("idf.npy", idf)

# =========================
# SVM（升级）
# =========================
clf = SVC(kernel='rbf', C=10, gamma='scale')

clf.fit(X, train_labels)

pickle.dump(clf, open("svm_model.pkl", "wb"))

print("训练完成，模型已保存")