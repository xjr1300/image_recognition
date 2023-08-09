import os
import copy
import random
from collections import deque
from typing import Callable, List, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from image_recognition import cifar_dir, params_dir


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
    """
    print(f"len(data): {len(data)}")  # 50,000
    print(
        f"data[0].shape: {data[0].shape}"
    )  # torch.Size([3, 32, 32]) - 3チャネル、32x32ピクセル画像
    """
    data = torch.stack(data)
    # print(f"data.shape: {data.shape}")  # torch.Size([50000, 3, 32, 32])
    # 各チャネルの平均値と標準偏差を計算
    channel_mean = data.mean(dim=(0, 2, 3))
    # print(f"channel_mean.shape: {channel_mean.shape}")  # torch.Size([3])   - 3チャンネルの平均値
    channel_std = data.std(dim=(0, 2, 3))  # torch.Size([3])   - 3チャンネルの標準偏差
    # print(f"channel_std.shape: {channel_std.shape}")
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


class BasicBlock(nn.Module):
    """ResNet18の基本ブロック"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """イニシャライザ

        Args:
            in_channels (int): 入力チャネル数
            out_channels (int): 出力チャネル数
            stride (int, optional): ストライド数、デフォルトは1
        """
        super().__init__()

        """
        残差接続
        """
        # バッチ正規化でバイアスパラメータを設定するため、1つ目の畳み込み演算にバイアスを使用しない
        # ストライドが1より大きい場合は、出力される特徴マップが縮小
        # ストライドが1の場合、パディングに1を指定した特徴マップのサイズを維持
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        # チャネル別に標準化する処理
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 2つ目の畳み込み演算では、パディングに1を設定して特徴マップのサイズを維持
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        # チャネル別に標準化する処理
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ReLU活性化関数
        self.relu = nn.ReLU(inplace=True)

        # ストライドが1より大きい場合は、スキップ接続と残差接続から得られる特徴マップの高さと幅が
        # 異なるため、特徴マップの高さを幅を合わせるために、スキップ接続から得られる特徴誤差に適用する
        # 畳み込み演算を用意
        self.downsample = None
        if 1 < stride:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def residual_connection(self, x: torch.Tensor) -> torch.Tensor:
        """残差接続させる。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, チャネル数, 高さ, 幅]
        Returns:
            torch.Tensor: 出力[バッチサイズ, チャネル数, 高さ, 幅]
        """
        # 1つ目の畳み込み演算
        out = self.conv1(x)
        # バッチ正規化
        out = self.bn1(out)
        # 活性化関数
        out = self.relu(out)
        # 2つ目の畳み込み演算
        out = self.conv2(out)
        # バッチ正規化
        return self.bn2(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播させる。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, 入力チャネル数, 高さ, 幅]
        Returns:
            torch.Tensor: 出力[バッチサイズ, 出力チャネル数, 高さ, 幅]
        """
        # 残差接続
        out = self.residual_connection(x)
        # スキップ接続を加えるため残差接続の特徴マップを縮小
        if self.downsample is not None:
            x = self.downsample(x)
        # 残差写像と恒等写像の要素ごとの和を計算
        out += x
        # ReLU活性化関数を適用
        return self.relu(out)


class ResNet18(nn.Module):
    """ResNet18モデル"""

    def __init__(
        self,
        num_classes: int,
        improved: bool = False,
    ) -> None:
        """イニシャライザー

        Args:
            num_classes (int): 分類対象の物体クラス数
            should_improve (bool): 改良したResNet18モデルを使用するかを指定するフラグ、デフォルトはFalse
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )
        self.dropout = nn.Dropout() if improved else None
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)
        if improved:
            self.init_params()

    def init_params(self) -> None:
        """Heらが提案した正規分布を使用してパラメータを初期化する。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor, return_embed: bool = False) -> torch.Tensor:
        """順伝播させる。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, チャネル数, 高さ, 幅]
            return_embed (bool): 特徴量またはロジットを返すかを指定するフラグ
        Returns:
            torch.Tensor: 出力[バッチサイズ, クラス数]
        """
        # 入力層 -> 隠れ層
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        # 隠れ層 -> 隠れ層
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 隠れ層 -> 出力層
        out = self.avg_pool(out)
        out = out.flatten(1)
        if return_embed:
            return out
        out = self.linear(out)
        return out

    def get_device(self) -> torch.device:
        """モデルパラメーターが保持されているデバイスを返す。"""
        return self.linear.weight.device

    def copy(self) -> Self:
        """モデルのコピーを返す。"""
        return copy.deepcopy(self)


class CNNConfig:
    """ハイパーパラメータとオプション"""

    def __init__(self) -> None:
        """イニシャライザー"""
        self.val_ratio = 0.2  # 検証に使う学習セット内のデータの割合
        self.num_epochs = 30  # 学習エポック数
        self.lr_drop = 25  # 学習率を減衰させるエポック数
        self.lr = 1e-2  # 学習率
        self.moving_avg = 20  # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32  # バッチサイズ
        self.num_workers = 2  # データローダに使うCPUプロセスの数
        # self.device = "cuda"   # 学習にCUDA対応GPUを使用
        # self.device = "cpu"  # 学習にCPUを使用
        self.device = "mps"  # 学習にAppleシリコンGPUを使用(MPS: Metal Performance Shaders)
        self.num_samples = 200  # t-SNEでプロットするサンプル数


def cnn_train_transforms(channel_mean: float, channel_std: float) -> T.Compose:
    """CNNで画像変換する関数を返す。

    Args:
        channel_mean (float): 各チャネルの平均
        channel_std (float): 各チャネルの標準偏差

    Returns:
        T.Compose: 画像変換関数
    """

    return T.Compose(
        (
            T.ToTensor(),
            T.Normalize(mean=channel_mean, std=channel_std),
        )
    )


def cnn_expand_train_transforms(channel_mean: float, channel_std: float) -> T.Compose:
    """CNNで学習データを拡張するときに画像変換する関数を返す。

    Args:
        channel_mean (float): 各チャネルの平均
        channel_std (float): 各チャネルの標準偏差

    Returns:
        T.Compose: 画像変換関数
    """
    return T.Compose(
        (
            T.RandomResizedCrop(32, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=channel_mean, std=channel_std),
        )
    )


def cnn_train_evaluation(improved: bool = False):
    """CNNモデルを学習して評価する。

    Args:
        improved (bool): 改良したCNNモデルを学習するかを指定するフラグ、デフォルトはFalse
    """

    config = CNNConfig()

    # 入力データを正規化するために学習セットのデータを使用して、各チャネルの平均と標準偏差を計算
    # torchvision.transforms.ToTensorは、PILまたはnumpy.ndarray(H, W, C)を(C, H, W)に変換して、
    # 各値を[0, 1]の範囲に正規化
    dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=T.ToTensor()
    )
    channel_mean, channel_std = calculate_dataset_statistics_by_channels(dataset)
    print(f"channel_mean: {channel_mean}")
    print(f"channel_std: {channel_std}")

    # 画像を整形する関数を用意
    train_transforms = (
        cnn_expand_train_transforms(channel_mean, channel_std)
        if improved
        else cnn_train_transforms(channel_mean, channel_std)
    )
    test_transforms = cnn_train_transforms(channel_mean, channel_std)

    # 学習及び評価データセットを用意
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=train_transforms
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=test_transforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=False, download=True, transform=test_transforms
    )

    # 学習データセットを学習または評価データセットに分割するインデックス集合を生成
    val_indices, train_indices = generate_subset_indices_pair(
        len(train_dataset), config.val_ratio
    )
    print(f"学習セットのサンプル数: {len(train_indices)}")
    print(f"検証セットのサンプル数: {len(val_indices)}")
    print(f"テストセットのサンプル数: {len(test_dataset)}")

    # 学習データセットのインデックス集合から無作為にインデックスをサンプルするサンプラーを生成
    train_sampler = SubsetRandomSampler(train_indices)

    # データローダーを生成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
    )
    val_loader_dataset = val_dataset if improved else train_dataset
    val_loader = DataLoader(
        val_loader_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_indices,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # 目的関数を生成
    loss_func = F.cross_entropy

    # 検証データセットの結果による最良モデルを保存する変数
    val_loss_best = float("inf")
    model_best = None

    # ResNet18モデルを生成
    model = ResNet18(len(train_dataset.classes), improved)

    # モデルをデバイスに転送
    model.to(config.device)

    # 最適化器を生成
    optimizer = (
        optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-5)
        if improved
        else optim.SGD(model.parameters(), lr=config.lr)
    )
    # 学習率減衰を管理するスケジューラー
    scheduler = (
        optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[config.lr_drop], gamma=0.1
        )
        if improved
        else None
    )

    for epoch in range(config.num_epochs):
        # モジュールを学習モードに設定
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"[エポック {epoch + 1}")
            # 移動平均計算用変数の初期化
            losses = deque(maxlen=config.moving_avg)
            accuracies = deque(maxlen=config.moving_avg)
            for x, y in pbar:
                # データをモデルと同じデバイスに転送
                x = x.to(model.get_device())
                y = y.to(model.get_device())
                # パラメーターの勾配をリセット
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
                pbar.set_postfix(
                    {
                        "loss": torch.Tensor(losses).mean().item(),
                        "accuracy": torch.Tensor(accuracies).mean().item(),
                    }
                )
        # 検証データセットを使用して精度を評価
        val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
        print(f"検証: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")
        # 検証データセットの結果による最良モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()
        # エポック終了時に学習率を更新
        if scheduler is not None:
            scheduler.step()
    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f"テスト: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")
    # t-SNEを使用して特徴量の分布をプロットして可視化
    plot_t_sne(test_loader, model_best, config.num_samples)
    # 最良のモデルを保存
    filename = "resnet18_improved_params.pth" if improved else "resnet18_params.pth"
    path = os.path.join(params_dir(), filename)
    torch.save(model_best.state_dict(), path)
