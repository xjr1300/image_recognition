import os
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from image_recognition import cifar_dir


def generate_subset_indices_pair(
    dataset: Dataset, ratio: float, random_seed: int = 0
) -> Tuple[List[int], List[int]]:
    """データセットを2分割するインデックスを格納した2つのリストを返す。

    Args:
        dataset (Dataset): 2つに分割するデータセット
        ratio (float): 分割した最初のセットに含めるデータ数を示す割合
        random_seed: データセットをランダムに分割するための乱数を生成するシード値
    Returns:
        Tuple[List[int], List[int]]: それぞれ2つのサブセットに含めるデータセットに含まれるデータ
        のインデックスを格納したタプル
    """
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)
    indices = list(range(len(dataset)))
    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)
    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]
    return indices1, indices2


def transform(
    image: Image.Image, channel_mean: np.ndarray = None, channel_std: np.ndarray = None
) -> np.ndarray:
    """画像を正規化する。

    Args:
        image (Image.Image): 正規化する画像
        channel_mean (np.ndarray): 各次元のデータセット全体の平均(入力次元)
        channel_std (np.ndarray): 各次元のデータセット全体の標準偏差(入力次元)
    Returns:
        np.ndarray: 正規化された画像
    """
    # 画像をNumPy配列に変換
    # img.shape: (32, 32, 3)
    image = np.asarray(image, dtype="float32")
    # (32, 32. 3)の画像を3072次元のベクトルに変換
    # 3072 = 32 * 32 * 3
    x = image.flatten()
    # 各次元をデータセット全体の平均と標準偏差で正規化
    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
    return x


def calculate_dataset_statistics(dataset: Dataset) -> Tuple[float, float]:
    """各次元のデータセット全体の平均と標準偏差を計算する。

    Args:
        dataset (Dataset): 平均と標準偏差を計算するデータセット
    Returns:
        Tuple[float, float]: 平均と標準偏差
    """
    data = []
    for i in range(len(dataset)):
        img_flat = dataset[i][0]
        data.append(img_flat)
    # print("data[0].shape: ", data[0].shape)  # (3072,)
    # 第0軸を追加して、第0軸でデータを連結
    data = np.stack(data)
    # print(f"data.shape: {data.shape}")  # (50000, 3072)
    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)
    # print(f"channel_mean.shape: {channel_mean.shape}")  # (3072,)
    # print(f"channel_std.shape: {channel_std.shape}")  # (3072,)
    return channel_mean, channel_std


def target_transform(label: int, num_classes: int = 10) -> np.ndarray:
    """ラベルをone-hotベクトルに変換する。

    Args:
        label (int): one-hotベクトルに変換するラベル
        num_classes (int): ラベルの数
    Returns:
        np.ndarray: one-hotベクトル
    """
    # 数字 -> one-hotベクトル
    y = np.identity(num_classes)[label]
    # print(f"y.shape: {y.shape}")
    return y


class MultiClassLogisticRegression:
    """多クラスロジスティック回帰モデル"""

    def __init__(self, dim_input: int, num_classes: int) -> None:
        """イニシャライザ

        Args:
            dim_input (int): 入力次元数
            num_classes (int): 分類対象の物体クラスの数
        """
        # パラメータをランダムに初期化
        self.weight = np.random.normal(scale=0.01, size=(dim_input, num_classes))
        self.bias = np.zeros(num_classes)
        # print(f"weight.shape: {self.weight.shape}") (3072, 10)
        # print(f"bias.shape: {self.bias.shape}")  # (10,)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """ソフトマックス関数

        Args:
            x (np.ndarray): ソフトマックス関数の入力
        Returns:
            np.ndarray: ソフトマックス関数の出力
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """物体クラスの確立を予測する。

        Args:
            x (np.ndarray): 入力データ(バッチサイズ, 入力次元)
        Returns:
            np.ndarray: 物体クラスの確率(バッチサイズ, 物体クラスの数)
        """
        y = np.matmul(x, self.weight) + self.bias
        y = self._softmax(y)
        # print(f"y.shape: {y.shape}")

        return y

    def update_parameters(
        self, x: np.array, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.001
    ):
        # 出力と正解の誤差を計算
        diffs = y_pred - y
        # 勾配を使用してパラメーターを更新
        self.weight -= lr * np.mean(x[:, :, np.newaxis] * diffs[:, np.newaxis], axis=0)
        self.bias -= lr * np.mean(diffs, axis=0)

    def copy(self):
        """モデルを複製する。"""
        # model_copy = self.__class__(*self.weight.shape)
        model_copy = MultiClassLogisticRegression(*self.weight.shape)
        model_copy.weight = self.weight.copy()
        model_copy.bias = self.bias.copy()
        return model_copy


class Config:
    """ハイパーパラメーターとオプション"""

    def __init__(self) -> None:
        self.val_ratio = 0.2  # 検証に使う学習セット内のデータの割合
        self.num_epochs = 30  # 学習エポック数
        self.lrs = [1e-2, 1e-3, 1e-4]  # 検証する学習率
        self.moving_avg = 20  # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32  # バッチサイズ
        self.num_workers = 2  # データローダーに使うCPUプロセスの数


class ImageTransformer:
    def __init__(self, channel_mean: float, channel_std: float) -> None:
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __call__(self, image: Image.Image) -> np.ndarray:
        return transform(image, self.channel_mean, self.channel_std)


def evaluate_train_dataset():
    config = Config()
    # 入力データ正規化のために学習セットのデータを使って
    # 各次元の平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=transform
    )
    channel_mean, channel_std = calculate_dataset_statistics(dataset)

    # 正規化を含めた画像整形関数の用意
    # img_transform = lambda x: transform(x, channel_mean, channel_std)
    # pickleでモデルを保存するため、lambda式を使用できないためコメントアウト
    # 代わりにImageTransformerクラスを定義して使用
    # img_transform = img_transform_wrapper(channel_mean, channel_std)

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(),
        train=True,
        download=True,
        # transform=img_transform,
        transform=ImageTransformer(channel_mean, channel_std),
        target_transform=target_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(),
        train=False,
        download=True,
        # transform=img_transform,
        transform=ImageTransformer(channel_mean, channel_std),
        target_transform=target_transform,
    )
    # 学習・検証セットへ分割するためのインデックス集合の生成
    val_set, train_set = generate_subset_indices_pair(train_dataset, config.val_ratio)
    print(f"学習セットのサンプル数: {len(train_set)}")  # 40000
    print(f"検証セットのサンプル数: {len(val_set)}")  # 10000
    print(f"テストセットのサンプル数: {len(test_dataset)}")  # 10000

    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_set)
    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_set,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float("inf")
    model_best = None
    for lr in config.lrs:
        # 多クラスロジスティック回帰モデルの生成
        print(f"学習率: {lr}")  # 0.01, 0.001, 0.0001
        model = MultiClassLogisticRegression(32 * 32 * 3, len(train_dataset.classes))
        for epoch in range(config.num_epochs):
            with tqdm(train_loader) as pbar:
                pbar.set_description(f"[エポック {epoch + 1}]")
                # 移動平均計算用
                losses = deque()
                accuracies = deque()
                for x, y in pbar:
                    # サンプルしたデータはPyTorchのTensorに
                    # 変換されているのためNumPyデータに戻す
                    x = x.numpy()
                    y = y.numpy()
                    y_pred = model.predict(x)
                    # 学習データに対する目的関数と正確度を計算
                    loss = np.mean(np.sum(-y * np.log(y_pred), axis=1))
                    # Maxのインデックスは数字表現のクラスラベル
                    accuracy = np.mean(
                        np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
                    )
                    # 移動平均を計算して表示
                    losses.append(loss)
                    accuracies.append(accuracy)
                    if len(losses) > config.moving_avg:
                        losses.popleft()
                        accuracies.popleft()
                    pbar.set_postfix(
                        {"loss": np.mean(losses), "accuracy": np.mean(accuracies)}
                    )
                    # パラメータを更新
                    model.update_parameters(x, y, y_pred, lr=lr)
            # 検証セットを使って精度評価
            val_loss, val_accuracy = evaluate(val_loader, model)
            print(f"検証: loss = {val_loss:.3f}, " f"accuracy = {val_accuracy:.3f}")
            # より良い検証結果が得られた場合、モデルを記録
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_best = model.copy()
    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best)
    print(f"テスト: loss = {test_loss:.3f}, " f"accuracy = {test_accuracy:.3f}")


def evaluate(data_loader: DataLoader, model: MultiClassLogisticRegression):
    losses = []
    preds = []
    for x, y in data_loader:
        x = x.numpy()
        y = y.numpy()
        y_pred = model.predict(x)
        losses.append(np.sum(-y * np.log(y_pred), axis=1))
        preds.append(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    loss = np.mean(np.concatenate(losses))
    accuracy = np.mean(np.concatenate(preds))
    return loss, accuracy


# `if __name__ == "__main__":`を追加
# この条件を満たすときにのみ、train_eval関数を呼び出さないと、multiprocessingでRuntimeErrorが発生する。
# RuntimeError:
#       An attempt has been made to start a new process before the
#       current process has finished its bootstrapping phase.
#
#       This probably means that you are not using fork to start your
#       child processes and you have forgotten to use the proper idiom
#       in the main module:
#
#           if __name__ == '__main__':
#               freeze_support()
#               ...
#
#       The "freeze_support()" line can be omitted if the program
#       is not going to be frozen to produce an executable.
if __name__ == "__main__":
    evaluate_train_dataset()
