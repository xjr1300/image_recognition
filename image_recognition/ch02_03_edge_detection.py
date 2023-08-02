import os

from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from image_recognition import image_resource_dir
from image_recognition.filters import edge_kernels

kernel_h, kernel_v, kernel_lap = edge_kernels()
print(f"水平方向のエッジ検出カーネル:\n{kernel_h}")
print(f"垂直方向のエッジ検出カーネル:\n{kernel_v}")
print(f"ラプラシアンフィルタカーネル:\n{kernel_lap}")

# グレースケール画像を開く
image_path = os.path.join(image_resource_dir(), "coffee.jpg")
image = Image.open(image_path)
# グレースケール画像をNumPy配列に変換
# 次に実行する畳み込み演算の結果は8bit符号なし整数の範囲を超えることがあるため4バイト整数に変換
image = np.array(image, dtype="int32")

# 水平方向のエッジを検出する畳み込み演算
image_h_diff = signal.convolve2d(image, kernel_h, mode="same")
# 垂直方向のエッジを検出する畳み込み演算
image_v_diff = signal.convolve2d(image, kernel_v, mode="same")
# ラプラシアンフィルタを適用
image_lap = signal.convolve2d(image, kernel_lap, mode="same")

# 微分値の絶対値を取得
image_h_diff = np.absolute(image_h_diff)
image_v_diff = np.absolute(image_v_diff)

# 水平一次微分画像と垂直一次微分画像を合成
image_diff = (image_h_diff**2 + image_v_diff**2) ** 0.5

# 最小値が0、最大値が255になるように画像を正規化
image_h_diff = np.clip(image_h_diff, 0, 255).astype("uint8")
image_v_diff = np.clip(image_v_diff, 0, 255).astype("uint8")
image_diff = np.clip(image_diff, 0, 255).astype("uint8")
image_lap = np.clip(image_lap, 0, 255).astype("uint8")

# イメージを表示
plt.figure(num="第2章 第3節 エッジ検出", figsize=(10, 7))
# オリジナル画像
plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
# 水平一次微分画像
plt.subplot(2, 3, 2)
plt.imshow(image_h_diff, cmap="gray")
plt.title("Horizontal Difference")
# 垂直一次微分画像
plt.subplot(2, 3, 3)
plt.imshow(image_v_diff, cmap="gray")
plt.title("Vertical Difference")
# 水平一次微分画像と垂直一次微分画像を合成した画像
plt.subplot(2, 3, 4)
plt.imshow(image_diff, cmap="gray")
plt.title("Horizontal and Vertical Difference")
# ラプラシアンフィルタを適用した画像
plt.subplot(2, 3, 5)
plt.imshow(image_lap, cmap="gray")
plt.title("Laplacian Filter")
plt.show()
