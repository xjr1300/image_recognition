import copy
from collections import deque
from typing import Self

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from image_recognition import cifar_dir
from image_recognition.utils import (
    calculate_dataset_statistics_by_channels,
    evaluate,
    generate_subset_indices_pair,
    plot_t_sne,
)


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

    def __init__(self, num_classes: int) -> None:
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

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


class Config:
    """ハイパーパラメータとオプション"""

    def __init__(self) -> None:
        """イニシャライザー"""
        self.val_ratio = 0.2  # 検証に使う学習セット内のデータの割合
        self.num_epochs = 30  # 学習エポック数
        self.lr = 1e-2  # 学習率
        self.moving_avg = 20  # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32  # バッチサイズ
        self.num_workers = 2  # データローダに使うCPUプロセスの数
        # self.device = "cuda"   # 学習にCUDA対応GPUを使用
        # self.device = "cpu"  # 学習にCPUを使用
        self.device = "mps"  # 学習にAppleシリコンGPUを使用(MPS: Metal Performance Shaders)
        self.num_samples = 200  # t-SNEでプロットするサンプル数


def train_eval():
    """学習及び評価する。"""

    config = Config()

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
    transforms = T.Compose(
        (
            T.ToTensor(),
            T.Normalize(mean=channel_mean, std=channel_std),
        )
    )

    # 学習及び評価データセットを用意
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=transforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=False, download=True, transform=transforms
    )

    # 学習データセットを学習または評価データセットに分割するインデックス集合を生成
    val_indices, train_indices = generate_subset_indices_pair(
        len(train_dataset), config.val_ratio
    )

    # 学習データセットのインデックス集合から無作為にインデックスをサンプルするサンプラーを生成
    train_sampler = SubsetRandomSampler(train_indices)

    # データローダーを生成
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

    # 検証データセットの結果による最良モデルを保存する変数
    val_loss_best = float("inf")
    model_best = None

    # ResNet18モデルを生成
    model = ResNet18(len(train_dataset.classes))

    # モデルをデバイスに転送
    model.to(config.device)

    # 最適化器を生成
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        # モジュールを学習モードに設定
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"[エポック {epoch + 1}")
            # 移動平均計算用変数の初期化
            # losses = deque()
            # accuracies = deque()
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
                # if config.moving_avg < len(losses):
                #     losses.popleft()
                #     accuracies.popleft()
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
    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f"テスト: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")
    # t-SNEを使用して特徴量の分布をプロットして可視化
    plot_t_sne(test_loader, model_best, config.num_samples)


if __name__ == "__main__":
    train_eval()
