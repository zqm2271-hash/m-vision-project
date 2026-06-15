# main.py

import os
import cv2

from utils import load_images, resize_to_same
from image_fusion import fuse_images


def main():

    # 图片路径
    img_a_path = "D:/lyxxx/1/picture/cat.png"
    img_b_path = "D:/lyxxx/1/picture/dog.png"

    # 读取图片
    img_a, img_b = load_images(
        img_a_path,
        img_b_path
    )

    # 统一尺寸
    img_a, img_b = resize_to_same(
        img_a,
        img_b
    )

    # 高频A + 低频B
    high_freq, low_freq, fused = fuse_images(
        img_a,
        img_b,
        kernel_size=31
    )

    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 保存结果
    cv2.imwrite(
        "output/high_frequency.png",
        high_freq
    )

    cv2.imwrite(
        "output/low_frequency.png",
        low_freq
    )

    cv2.imwrite(
        "output/fused.png",
        fused
    )

    print("处理完成")
    print("结果已保存到 output 文件夹")

    # 显示图片
    cv2.imshow("Image A", img_a)
    cv2.imshow("Image B", img_b)

    cv2.imshow("High Frequency", high_freq)
    cv2.imshow("Low Frequency", low_freq)

    cv2.imshow("Fused Result", fused)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()