import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from image_recognition import image_resource_dir

# グレースケール画像の読み込み
gray_image_path = os.path.join(image_resource_dir(), "coffee.jpg")
gray_image = Image.open(gray_image_path)
gray_image = np.array(gray_image)
print(f"グレースケール画像を保存する配列の形: {gray_image.shape}")
# x, y = 0, 0
x, y = 236, 191
print(f"グレースケール画像の({x}, {y})における画素値: {gray_image[x, y]}")

# カラー画像の読み込み
color_image_path = os.path.join(image_resource_dir(), "apple.jpg")
color_image = Image.open(color_image_path)
color_image = np.array(color_image)
print(f"カラー画像を保存する配列の形: {color_image.shape}")
x, y = 236, 191
print(f"カラー画像の({x}, {y})における画素値: {color_image[x, y]}")

# 2つの画像を並べて表示
plt.figure(num="第2章 第1節 画像データを読み込んで表示", figsize=(10, 7))
plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Gray Image")
plt.subplot(2, 3, 2)
plt.imshow(color_image)
plt.title("Color Image")
# カラー画像をチャンネル別に並べて表示
plt.subplot(2, 3, 4)
plt.imshow(color_image[:, :, 0], cmap="gray")
plt.title("Red Channel")
plt.subplot(2, 3, 5)
plt.imshow(color_image[:, :, 1], cmap="gray")
plt.title("Green Channel")
plt.subplot(2, 3, 6)
plt.imshow(color_image[:, :, 2], cmap="gray")
plt.title("Blue Channel")
plt.show()
