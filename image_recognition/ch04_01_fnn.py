"""FNN(Feedforward Neural Network)"""

import copy
from collections import deque
from typing import Callable, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from image_recognition import cifar_dir
from image_recognition.utils import (
    calculate_dataset_statistics,
    generate_subset_indices_pair,
    transform,
)


def generate_hidden_layer(dim_input: int, dim_output: int) -> nn.Sequential:
    """隠れ層を生成する。

    Args:
        dim_input (int): 入力の次元数
        dim_output (int): 出力の次元数
    Returns:
        nn.Sequential: 隠れ層
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_output, bias=False),
        nn.BatchNorm1d(dim_output),
        nn.ReLU(inplace=True),
    )


class FNN(nn.Module):
    """順伝播型ニューラルネットワーク

    torch.nn.Moduleは、ニューラルネットワークのモジュールの基本クラスである。
    独自モジュールは、torch.nn.Moduleを継承し、forwardメソッドを実装することで作成できる。
    モジュールには他のモジュールを含めることもでき、モジュールをツリー構造に入れ子にすることができる。
    サブモジュールを通常の属性として割り当てることもできる。
    """

    def __init__(
        self, dim_input: int, dim_hidden: int, num_hidden_layers: int, num_classes: int
    ) -> None:
        """イニシャライザ

        Args:
            dim_input (int): 入力の次元数
            dim_hidden (int): 特徴量の次元数
            num_hidden_layers (int): 隠れ層の数
            num_classes (int): 分類対象の物体クラスの数
        """
        super().__init__()
        self.layers = nn.ModuleList()
        # 入力層 -> 隠れ層
        self.layers.append(generate_hidden_layer(dim_input, dim_hidden))
        # 隠れ層 -> 隠れ層
        for _ in range(num_hidden_layers - 1):
            self.layers.append(generate_hidden_layer(dim_hidden, dim_hidden))
        # 隠れ層 -> 出力層
        self.linear = nn.Linear(dim_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_embed: bool = False) -> torch.Tensor:
        """順伝播させる。

        ロジットとは、ソフトマックス活性化関数を適用する前のNNの出力を示す。
        ロジットは、入力特徴量と重みの線形結合を計算することで得られ、NNの学習において、適切な重みとバイアスが学習されて
        最終的な予測を行うための情報源として機能する。

        Args:
            x (torch.Tensor): 入力
            return_embed (bool): 特徴量を返すか、ロジットを返すかを示すフラグ
        Returns:
            torch.Tensor: 特徴量またはロジット
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        if return_embed:
            # 特徴量を返す
            return h
        # ロジットを返す
        return self.linear(h)

    def get_device(self) -> torch.device:
        """デバイスを取得する。

        Returns:
            torch.device: デバイス
        """
        return self.linear.weight.device

    def copy(self) -> Self:
        """モデルをディープコピーする。

        Returns:
            Self: ディープコピーされたモデル
        """
        return copy.deepcopy(self)


class Config:
    """ハイパーパラメータとオプションを保持するクラス"""

    def __init__(self) -> None:
        self.val_ratio = 0.2  # 検証に使う学習セット内のデータの割合
        self.dim_hidden = 512  # 隠れ層の特徴量次元
        self.num_hidden_layers = 2  # 隠れ層の数
        self.num_epochs = 30  # 学習エポック数
        self.lr = 1e-2  # 学習率
        self.moving_avg = 20  # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32  # バッチサイズ
        self.num_workers = 2  # データローダに使うCPUプロセスの数
        self.device = "cpu"  # 学習に使うデバイス
        self.num_samples = 200  # t-SNEでプロットするサンプル数


class ImageTransformer:
    def __init__(self, channel_mean: np.ndarray, channel_std: np.ndarray) -> None:
        """イニシャライザ

        Args:
            channel_mean (np.ndarray): 各チャネルの平均
            channel_std (np.ndarray): 各チャネルの標準偏差
        """
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __call__(self, image: Image.Image) -> np.ndarray:
        """画像を正規化する。

        Args:
            image (Image.Image): 画像
        Returns:
            np.ndarray: 正規化された画像
        """
        return transform(image, self.channel_mean, self.channel_std)


def evaluate_train_subset():
    config = Config()
    # 入力データを正規化するために、学習データセットを使用して、各次元の平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=transform
    )
    # 学習データセットの平均と標準偏差を計算
    channel_mean, channel_std = calculate_dataset_statistics(dataset)
    # 学習及びテストデータセットを準備
    image_transformer = ImageTransformer(channel_mean, channel_std)
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=image_transformer
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=False, download=True, transform=image_transformer
    )
    # 学習データセットを学習サブセットと検証サブセットに分割
    train_indices, val_indices = generate_subset_indices_pair(
        len(train_dataset), config.val_ratio
    )
    print(f"学習サブセットのサンプル数: {len(train_indices)}")
    print(f"検証サブセットのサンプル数: {len(val_indices)}")
    print(f"テストデータセットのサンプル数: {len(test_dataset)}")

    # インデックス集合から無作為にインデックスを抽出するサンプラーを作成
    # torch.utils.data.SubsetRandomSampler(indices, generator=None)
    # 与えられたインデックスのリストから、インデックスを置き換えしないでランダムに要素をサンプリングする。
    # indices (list): サンプルするインデックスのリスト
    # generator (torch.Generator, optional): サンプルするインデックスを生成するためのジェネレータ
    train_sampler = SubsetRandomSampler(train_indices)

    # データローダーを生成
    # torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None,
    #     num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
    #     worker_init_fn=None, multiprocessing_context=None, generator=None, *,
    #     prefetch_factor=None, persistent_workers=False, pin_memory_device=''
    # )
    # データセットとサンプラーを結合して、与えられたデータセットを反復処理するためのイテレータを提供する。
    #
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
        sampler=val_indices,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # 目的関数を生成
    loss_func = F.cross_entropy
    # 検証サブセットの結果により、最良のモデルを保存する変数を初期化
    val_loss_best = float("inf")
    model_best = None

    # FNNモデルを生成
    model = FNN(
        32 * 32 * 3,
        config.dim_hidden,
        config.num_hidden_layers,
        len(train_dataset.classes),
    )
    # モデルをデフォルトのCPUデバイスに転送
    model.to(config.device)
    # 最適化器を生成
    # torch.optim.SGD(
    #     params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0,
    #     nesterov=False, *, maximize=False, lr_in_momentum=False, differentiable_params=None
    # )
    # 確立的勾配降下法を実装した最適化アルゴリズム
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()
        # 学習サブセットのデータを反復処理
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"Epoch {epoch + 1}")
            # 移動平均計算用キューを初期化
            losses = deque()
            accuracies = deque()
            counter = 0
            for x, y in pbar:
                # データをモデルと同じデバイスに転送
                # x.shape: torch.Size([32, 3072])
                # y.shape: torch.Size([32])
                x = x.to(model.get_device())
                y = y.to(model.get_device())
                # すでに計算された勾配をリセット
                optimizer.zero_grad()
                # 順伝播
                y_pred = model(x)
                # 学習データに対する損失と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()
                # 誤差逆伝播
                loss.backward()
                # パラメーターを更新
                optimizer.step()
                # 移動平均を計算して表示
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accuracies.popleft()
                counter += 1
                pbar.set_postfix(
                    {
                        "loss": torch.Tensor(losses).mean().item(),
                        "accuracy": torch.Tensor(accuracies).mean().item(),
                        "counter": counter,
                    }
                )
        # 検証セットを使用して精度を評価
        val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
        print(f"検証: loss = {val_loss:.3f}, ", f"accuracy = {val_accuracy:.3f}")
        # より良い検証結果が得られた場合モデルを更新
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()
    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f"テスト: loss = {test_loss:.3f}, ", f"accuracy = {test_accuracy:.3f}")
    # t-SNEを使用して特徴量の分布をプロット
    plot_t_sne(test_loader, model_best, config.num_samples)


def evaluate(
    data_loader: DataLoader, model: nn.Module, loss_func: Callable
) -> Tuple[float, float]:
    """評価する。

    Args:
        data_loader (DataLoader): データローダー
        model (nn.Module): モデル
        loss_func (Callable): 損失関数
    Returns:
        Tuple[float float]: 損失と正確度
    """
    model.eval()
    losses = []
    preds = []
    for x, y in data_loader:
        x = x.to(model.get_device())
        y = y.to(model.get_device())
        y_pred = model(x)
        losses.append(loss_func(y_pred, y, reduction="none"))
        preds.append(y_pred.argmax(dim=1) == y)
    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()
    return loss, accuracy


def plot_t_sne(data_loader: DataLoader, model: nn.Module, num_samples: int):
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


if __name__ == "__main__":
    evaluate_train_subset()
