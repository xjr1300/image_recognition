import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from image_recognition import image_resource_dir
from image_recognition.filters import gaussian_kernel
from image_recognition.processing import apply_convolution

image_path = os.path.join(image_resource_dir(), "coffee_noise.jpg")
image = Image.open(image_path)
kernel_size = 5

# ガウシアンカーネルを生成して、画像に適用
kernel = gaussian_kernel(width=kernel_size, height=kernel_size, sigma=1.3)
print(f"ガウシアンカーネル:\n{kernel}")
gaussian_image = apply_convolution(image, kernel)

# メディアンフィルタを画像に適用
image = np.array(image)
median_image = cv2.medianBlur(image, ksize=kernel_size)
# バイラテラルフィルタを画像に適用
bilateral_image = cv2.bilateralFilter(
    image, d=kernel_size, sigmaColor=10, sigmaSpace=10
)

# ガウシアンフィルタを適用した結果を表示
plt.figure(num="第2章 第2節 ガウシアンフィルタを適用", figsize=(10, 7))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.subplot(2, 2, 2)
plt.imshow(gaussian_image, cmap="gray")
plt.title("Apply Gaussian Filter")
plt.subplot(2, 2, 3)
plt.imshow(median_image, cmap="gray")
plt.title("Apply Median Filter")
plt.subplot(2, 2, 4)
plt.imshow(bilateral_image, cmap="gray")
plt.title("Apply Bilateral Filter")
plt.show()
