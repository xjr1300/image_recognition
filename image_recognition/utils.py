import random
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import Dataset


def generate_subset_indices_pair(
    num_dataset: int, ratio: float, random_seed: int = 0
) -> Tuple[List[int], List[int]]:
    """データセットを2つに分割するときに、それぞれのサブセットに含まれるデータセット内のデータのインデックスを格納したタプルを返す。

    Args:
        num_dataset (int): データセット内のデータ数
        ratio (float): 分割するときの比率
        random_seed (int, optional): 乱数生成器のシード値
    Returns:
        Tuple[List[int], List[int]]: サブセットに含まれるデータセット内のデータのインデックスを格納したタプル
    """
    first_size = int(num_dataset * ratio)
    indices = list(range(num_dataset))
    random.seed(random_seed)
    random.shuffle(indices)
    return indices[:first_size], indices[first_size:]


def calculate_dataset_statistics(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """データセット内のデータの平均値と標準偏差を計算する。

    Args:
        dataset (Dataset): データセット
    Returns:
        Tuple[np.ndarray, np.ndarray]: データセット内のデータの平均値と標準偏差
    """
    data = np.stack([np.array(image).flatten() for image, _ in dataset])
    return data.mean(axis=0), data.std(axis=0)


def transform(
    image: Image.Image, channel_mean: np.ndarray = None, channel_std: np.ndarray = None
) -> np.ndarray:
    """イメージを特徴量に変換する。

    Args:
        image (Image.Image): イメージ
        channel_mean (np.ndarray, optional): チャネルごとの平均値
        channel_std (np.ndarray, optional): チャネルごとの標準偏差
    Returns:
        np.ndarray: 変換後の特徴量
    """
    # PILのイメージをNumPy配列に変換
    x = np.asarray(image, dtype="float32")
    # CIFAR10の場合、画像は[32, 32, 3]の形状
    # [32, 32, 3]の画像配列を3072次元の特徴量ベクトルに変換
    # 3072 = 32 * 32 * 3
    x = x.flatten()
    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
    return x


def calculate_dataset_statistics_by_channels(
    dataset: Dataset,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """データセット全体の平均と標準偏差をチャネル別に計算する。

    Args:
        dataset (Dataset): データセット
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: データセット全体のチャネル別の平均と標準偏差
    """
    data = [dataset[i][0] for i in range(len(dataset))]
    print(f"len(data): {len(data)}")  # 50,000
    print(
        f"data[0].shape: {data[0].shape}"
    )  # torch.Size([3, 32, 32]) - 3チャネル、32x32ピクセル画像
    data = torch.stack(data)
    print(f"data.shape: {data.shape}")  # torch.Size([50000, 3, 32, 32])
    # 各チャネルの平均値と標準偏差を計算
    channel_mean = data.mean(dim=(0, 2, 3))
    print(f"channel_mean.shape: {channel_mean.shape}")  # torch.Size([3])   - 3チャンネルの平均値
    channel_std = data.std(dim=(0, 2, 3))  # torch.Size([3])   - 3チャンネルの標準偏差
    print(f"channel_std.shape: {channel_std.shape}")
    return channel_mean, channel_std


def evaluate(
    data_loader: Dataset, model: nn.Module, loss_func: Callable
) -> Tuple[float, float]:
    model.eval()
    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())
            y_pred = model(x)
            losses.append(loss_func(y_pred, y, reduction="none"))
            preds.append(y_pred.argmax(dim=1) == y)
    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()
    return loss, accuracy


def plot_t_sne(data_loader: Dataset, model: nn.Module, num_samples: int):
    model.eval()
    # t-SNEのためにデータを整形
    x = []
    y = []
    for imgs, labels in data_loader:
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            # 特徴量の抽出
            embeddings = model(imgs, return_embed=True)
            x.append(embeddings.to("cpu"))
            y.append(labels.clone())
    x = torch.cat(x)
    y = torch.cat(y)
    # NumPy配列に変換
    x = x.numpy()
    y = y.numpy()
    # 指定サンプル数だけ抽出
    x = x[:num_samples]
    y = y[:num_samples]
    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)
    # 各ラベルの色とマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ["4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
        plt.scatter(
            x_reduced[y == i, 0],
            x_reduced[y == i, 1],
            c=[cmap(i / len(data_loader.dataset.classes))],
            marker=markers[i],
            s=500,
            alpha=0.6,
            label=cls,
        )
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
    plt.show()
