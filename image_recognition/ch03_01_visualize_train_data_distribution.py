import os

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from image_recognition import cifar_dir

# CIFAR-10のデータセットをダウンロード
# データセットがダウンロード済みの場合は、ダウンロードをスキップ
dataset = torchvision.datasets.CIFAR10(root=cifar_dir(), train=True, download=True)

# それぞれの物体クラスのすべてのラベルの画像を１枚ずつ表示するまでループ
plt.figure(num="CIFAR-10", figsize=(10, 10))
plt.subplots_adjust(left=0.05, right=0.995, bottom=0.05, top=0.995)
displayed_classes = set()
i = 0
num_classes = len(dataset.classes)
num_datasets = len(dataset)
while i < num_datasets and len(displayed_classes) < num_classes:
    image, label = dataset[i]
    if label not in displayed_classes:
        print(f"物体クラス: {dataset.classes[label]}")
        image = image.resize((256, 256))
        plt.subplot(3, 4, len(displayed_classes) + 1)
        plt.imshow(image)
        plt.title(dataset.classes[label])
        displayed_classes.add(label)
    i += 1
plt.show()

# t-SNEを使用するためにデータを整形
x = []
y = []
num_samples = 200
for i in range(num_samples):
    image, label = dataset[i]
    # 画像を平坦化([32, 32, 3], [3027]に変換)
    flatted = np.array(image).flatten()
    x.append(flatted)
    y.append(label)

# すべてのデータをNumPy配列に変換
x = np.stack(x)  # shape: (200, 3072)
y = np.array(y)  # shape: (200, )

# t-SNEを適用
t_sne = TSNE(n_components=2, random_state=0)
x_reduced = t_sne.fit_transform(x)

# 各ラベルの色とマーカーを設定
cmap = plt.get_cmap("tab10")
markers = ["4", "8", "2", "p", "*", "h", "H", "+", "x", "D"]

# データをプロット
plt.figure(num="CIFAR-10 t-SNE", figsize=(16, 10))
for i, cls in enumerate(dataset.classes):
    plt.scatter(
        x_reduced[y == i, 0],
        x_reduced[y == i, 1],
        c=[cmap(i / num_classes)],
        marker=markers[i],
        s=500,
        alpha=0.6,
        label=cls,
    )
plt.axis("off")
plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
plt.show()
