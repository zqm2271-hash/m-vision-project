# image_fusion.py

import cv2
import numpy as np


def extract_low_frequency(img, kernel_size=31):

    low_freq = cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        0
    )

    return low_freq


def extract_high_frequency(img, kernel_size=31):

    low_freq = cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        0
    )

    high_freq = cv2.subtract(img, low_freq)

    return high_freq


def fuse_images(img_a, img_b, kernel_size=31):

    # A提取高频
    high_freq = extract_high_frequency(
        img_a,
        kernel_size
    )

    # B提取低频
    low_freq = extract_low_frequency(
        img_b,
        kernel_size
    )

    # 融合
    fused = cv2.add(
        high_freq,
        low_freq
    )

    return high_freq, low_freq, fused