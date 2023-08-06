import random
from typing import List, Tuple

import numpy as np
from PIL import Image
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
