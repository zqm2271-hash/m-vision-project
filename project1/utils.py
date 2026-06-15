# utils.py

import cv2


def load_images(path_a, path_b):
    """
    读取两张图片
    """
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        raise ValueError(f"无法读取 {path_a}")

    if img_b is None:
        raise ValueError(f"无法读取 {path_b}")

    return img_a, img_b


def resize_to_same(img_a, img_b):
    """
    将两张图片统一尺寸
    以A图尺寸为基准
    """

    h, w = img_a.shape[:2]

    img_b = cv2.resize(
        img_b,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    return img_a, img_b