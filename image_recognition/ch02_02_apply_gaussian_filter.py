import os

from PIL import Image
from matplotlib import pyplot as plt

from image_recognition import image_resource_dir
from image_recognition.filters import gaussian_kernel
from image_recognition.processing import apply_convolution

image_path = os.path.join(image_resource_dir(), "coffee_noise.jpg")
image = Image.open(image_path)

# ガウシアンカーネルを生成して、画像に適用
kernel = gaussian_kernel(width=5, height=5, sigma=1.3)
print(f"ガウシアンカーネル:\n{kernel}")
filtered_image = apply_convolution(image, kernel)

# ガウシアンフィルタを適用した結果を表示
plt.figure(num="第2章 第2節 ガウシアンフィルタを適用", figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap="gray")
plt.title("Apply Gaussian Filter")
plt.show()
