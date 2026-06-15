import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from extract_features import extract_bovw_feature

test_paths = np.load("test_paths.npy", allow_pickle=True)
test_labels = np.load("test_labels.npy")

idf = np.load("idf.npy")
clf = pickle.load(open("svm_model.pkl", "rb"))

print("提取测试特征...")

X = []

for p in test_paths:
    x = extract_bovw_feature(p)
    x = x * idf
    X.append(x)

X = np.array(X)

pred = clf.predict(X)

acc = accuracy_score(test_labels, pred)

print("\n====================")
print("Accuracy:", acc * 100)
print("====================")