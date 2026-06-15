import cv2
import numpy as np
from scipy.spatial.distance import cdist

vocabulary = np.load("vocabulary.npy")
K = vocabulary.shape[0]

sift = cv2.SIFT_create()

def extract_bovw_feature(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        return np.zeros(K)

    kp, des = sift.detectAndCompute(img, None)

    if des is None:
        return np.zeros(K)

    dist = cdist(des, vocabulary)
    words = np.argmin(dist, axis=1)

    hist = np.bincount(words, minlength=K).astype(np.float32)

    # L2 normalization（重要升级）
    hist /= (np.linalg.norm(hist) + 1e-8)

    return hist