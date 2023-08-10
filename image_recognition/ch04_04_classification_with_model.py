import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image

from image_recognition import cifar_dir, image_resource_dir, params_dir
from image_recognition.utils import (
    CNNConfig,
    ResNet18,
    calculate_dataset_statistics_by_channels,
)


def classification():
    config = CNNConfig()
    # 入力画像を正規化するためにCIFAR-10の学習セットを使用して、チャネルごとの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=T.ToTensor()
    )
    channel_mean, channel_std = calculate_dataset_statistics_by_channels(dataset)

    transforms = T.Compose(
        (
            T.ToTensor(),
            T.Normalize(mean=channel_mean, std=channel_std),
        )
    )

    # ResNet18モデルの生成とパラメータの読み込み
    model = ResNet18(len(dataset.classes))
    param_path = os.path.join(params_dir(), "resnet18_improved_params.pth")
    model.load_state_dict(torch.load(param_path))

    # モデルをデバイスに転送
    model.to(config.device)
    # モデルを評価モードに設定
    model.eval()

    # 画像を読み込んで物体分類
    plt.figure(num="Classification", figsize=(13, 6))
    image_dir = os.path.join(image_resource_dir(), "classification")
    for i, image_path in enumerate(Path(image_dir).glob("*.jpg")):
        # 画像を読み込み
        image = Image.open(image_path)
        # 画像を正規化
        image.resize((256, 256))
        normalized_image = transforms(image)
        # 物体分類予測
        normalized_image = normalized_image.unsqueeze(0)
        normalized_image = normalized_image.to(config.device)
        pred = model(normalized_image).argmax()
        actual = dataset.classes[pred]

        # 画像と予測結果を表示
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        title = f"expect: {image_path.stem}\nactual: {actual}"
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    classification()
