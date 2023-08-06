import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from image_recognition import image_resource_dir

original_image_path = os.path.join(image_resource_dir(), "cosmos.jpg")
original_image = Image.open(original_image_path)
image = np.array(original_image, dtype="float32")
print(f"image.shape: {image.shape}")  # (256, 256, 3)

# 画像を特徴量(3, 8)に射影
w = np.array(
    [
        [0.0065, -0.0045, -0.0018, 0.0075, 0.0095, 0.0075, -0.0026, 0.0022],
        [-0.0065, 0.0081, 0.0097, -0.0070, -0.0086, -0.0107, 0.0062, -0.0050],
        [0.0024, -0.0018, 0.0002, 0.0023, 0.0017, 0.0021, -0.0017, 0.0016],
    ]
)
features = np.matmul(image, w)
print(f"features.shape: {features.shape}")  # (256, 256, 8)

# アテンション計算用の特徴を画像から抽出
# (50, 50)は白色のコスモスが写っている座標
# (200, 200)はピンク色のコスモスが写っている座標
feature_white = features[50, 50]  # shape: (8,)
feature_pink = features[200, 200]  # shape: (8,)

# アテンションの計算
attention_white = np.matmul(features, feature_white)  # (256, 256)
attention_pink = np.matmul(features, feature_pink)  # (256, 256)

# ソフトマックスの計算
attention_white = np.exp(attention_white) / np.sum(np.exp(attention_white))
attention_pink = np.exp(attention_pink) / np.sum(np.exp(attention_pink))


def normalize(x: np.ndarray) -> np.ndarray:
    x_min = np.amin(x)
    x_max = np.amax(x)
    return (x - x_min) / (x_max - x_min)


# 表示用に最小値と最大値で0から1の範囲に正規化
attention_white = normalize(attention_white)
attention_pink = normalize(attention_pink)

# NumPy配列をグレースケール画像に変換
attention_white_image = Image.fromarray((attention_white * 255).astype("uint8"))
attention_pink_image = Image.fromarray((attention_pink * 255).astype("uint8"))

# イメージを表示
plt.figure(num="第2章 第4節 アテンション", figsize=(10, 7))
# オリジナル画像
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original")
# アテンション画像(白)
plt.subplot(1, 3, 2)
plt.imshow(attention_white_image, cmap="gray")
plt.title("Attention(White)")
# アテンション画像(ピンク)
plt.subplot(1, 3, 3)
plt.imshow(attention_pink_image, cmap="gray")
plt.title("Attention(Pink)")
# 画像を表示
plt.show()
