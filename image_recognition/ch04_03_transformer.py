import copy
from collections import deque
from typing import Self

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from image_recognition import cifar_dir
from image_recognition.utils import (
    calculate_dataset_statistics_by_channels,
    evaluate,
    generate_subset_indices_pair,
    plot_t_sne,
)


class SelfAttention(nn.Module):
    """自己アテンションクラス"""

    def __init__(self, dim_hidden: int, num_heads: int, qkv_bias: bool = False) -> None:
        """イニシャライザー

        Args:
            dim_hidden (int): 入力特徴量の次元数
            num_heads (int): マルチヘッドアテンションのヘッド数
            qkv_bias (bool, optional): クエリ、キー及び値を生成する全結合層にバイアスを与えるかを示すフラグ、デフォルトはFalse。
        """
        super().__init__()
        # 特徴量を各ヘッドで分割するため、特徴量の次元をヘッド数で割り切れるか確認
        assert dim_hidden % num_heads == 0
        self.num_heads = num_heads
        # ヘッドごとの特徴量の次元数
        dim_head = dim_hidden // num_heads
        # ソフトマックス関数に適用するスケール値
        self.scale = dim_head**-0.5
        # ヘッドごとにクエリ、キー及び値を生成する全結合層
        # pytorch.nn.Linearは、入力データに線形変換(y = x *A^T + b)を適用
        # dim_hiddenは入力次元数で、dim_hidden * 3は出力次元数
        # bias=Falseとすることで、バイアス項(b)を無効化
        # 自己アテンションクラスは、入力特徴量はクエリ、キー及び値で同じため出力次元数を入力特徴量の3倍に設定
        self.proj_in = nn.Linear(dim_hidden, dim_hidden * 3, bias=qkv_bias)
        # 各ヘッドから得られた特徴量を1つに結合する全結合層
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播する。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, 特徴量数, 特徴量次元数]
        Returns:
            torch.Tensor: 出力[バッチサイズ, 特徴量数, 特徴量次元数]
        """
        bs, ns = x.shape[:2]  # バッチサイズ, 特徴量数
        qkv = self.proj_in(x)  # qkv: クエリ、キー及び値

        # [バッチサイズ, 特徴量数, QKV, ヘッド数, ヘッドごとの特徴量次元数]
        # -1は他の次元数から推測することを示す
        qkv = qkv.view(bs, ns, 3, self.num_heads, -1)
        # [QKV, バッチサイズ, ヘッド数, 特徴量数, ヘッドごとの特徴量次元数]
        # pytorch.Tensor.permute(*dims)は、テンソルの次元を入れ替え
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # クエリ、キー及び値に分解
        # torch.Tensor.unbind(dim)は、指定した次元でテンソルを分割
        # 結果として、3つの[1, 4]次元のテンソルを取得
        # q, k, v[バッチサイズ, ヘッド数, 特徴量数, ヘッドごとの特徴量次元数]
        q, k, v = qkv.unbind(0)

        # クエリとキーの行列積とアテンションを計算（マスク不使用）
        # torch.Tensor.matmul(tensor)は、テンソルの行列積を計算する。
        # 入力テンソルが2軸より大きい軸数を持つ場合、最後の2軸で行列積を計算する。
        # attentionは[バッチサイズ, ヘッド数, 特徴量数, 特徴量数]次元
        # pytorch.Tensor.transpose(*dims)は、テンソルの次元を入れ替え
        # k.transpose(-2, -1)は、kの最後から2番目の次元と最後の次元を入れ替え
        # p[32, 8, 65, 64]
        # k[32, 8, 64, 65]
        # attention[32, 8, 65, 65]
        attention = q.matmul(k.transpose(-2, -1))
        attention = (attention * self.scale).softmax(dim=-1)

        # アテンションと値の行列積により値を収集
        # x[バッチサイズ, ヘッド数, 特徴量数, ヘッドごとの特徴量次元数]
        # x[32, 8, 65, 64]
        x = attention.matmul(v)

        # permute関数により
        #   [バッチサイズ, 特徴量数, ヘッド数, ヘッドの特徴量次元]
        # flatten関数により全てのヘッドから得られる特徴量を連結して、
        #   [バッチサイズ, 特徴量数, ヘッド数 * ヘッドの特徴量次元]
        #   [32, 65, 512]
        #   512 = 8 * 64
        x = x.permute(0, 2, 1, 3).flatten(2)
        # x[32, 65, 512]
        x = self.proj_out(x)
        return x


class FNN(nn.Module):
    """Transformerエンコーダー内のFNNクラス"""

    def __init__(self, dim_hidden: int, dim_feedforward: int) -> None:
        """イニシャライザー

        Args:
            dim_hidden (int): 入力特徴量の次元数
            dim_feedforward (int): 中間特徴量の次元数
        """
        super().__init__()
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播する。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, 特徴量数, 特徴量次元数]
        Returns:
            torch.Tensor: 出力[バッチサイズ, 特徴量数, 特徴量次元数]
        """
        # x[32, 65, 512]
        x = self.linear1(x)
        # x[32, 65, 512]
        x = self.activation(x)
        # x[32, 65, 512]
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformerエンコーダーレイヤクラス"""

    def __init__(self, dim_hidden: int, num_heads: int, dim_feedforward: int) -> None:
        """イニシャライザー

        Args:
            dim_hidden (int): 入力特徴量の次元数
            num_heads (int): マルチヘッドアテンションのヘッド数
            dim_feedforward (int): 中間特徴量の次元数
        """
        super().__init__()
        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播する。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, 特徴量数, 特徴量次元数]
        """
        # x[32, 65, 512]
        x = self.norm1(x)
        # x[32, 65, 512]
        x = self.attention(x) + x
        # x[32, 65, 512]
        x = self.norm2(x)
        # x[32, 65, 512]
        x = self.fnn(x) + x
        return x


class VisionTransformer(nn.Module):
    """Vision Transformerクラス"""

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        patch_size: int,
        dim_hidden: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        """イニシャライザー

        Args:
            num_classes (int): 分類対象の物体クラス数
            image_size (int): 画像サイズ(高さと幅が同じことを想定)
            patch_size (int): パッチサイズ(高さと幅が同じことを想定)
            dim_hidden (int): 入力特徴量の次元数
            num_heads (int): マルチヘッドアテンションのヘッド数
            dim_feedforward (int): FNNによる中間特徴量の次元数
            num_layers (int): Transformerコンコーダーのレイヤー数
        """
        assert image_size % patch_size == 0

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # パッチの行数と列数はimage_size // patch_size
        # パッチ数は(image_size // patch_size) ** 2
        num_patches = (image_size // patch_size) ** 2
        # パッチ特徴量はパッチを平坦化することで生成されるため、その次元数は
        # patch_size * patch_size * 3
        # 3はRGBの3チャネルを示す
        dim_patch = patch_size**2 * 3
        # パッチ特徴量をTransformerコンコーダーに入力する前に、パッチ特徴量の次元数を
        # 変換する全結合層
        self.patch_embed = nn.Linear(dim_patch, dim_hidden)
        # 位置埋め込み（バッチ数 + クラス埋め込みだけ用意）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim_hidden))
        # クラス埋め込み
        self.class_token = nn.Parameter(torch.zeros((1, 1, dim_hidden)))
        # Transformerコンコーダー層
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward)
                for _ in range(num_layers)
            ]
        )
        # ロジットを生成する前のレイヤ正規化と全結合
        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        """順伝播する。

        Args:
            x (torch.Tensor): 入力[バッチサイズ, チャネル数, 高さ, 幅]
            return_embed (bool, optional): 特徴量またはロジットを返すかを指定するフラグ
        """
        bs, c, h, w = x.shape
        # 入力画像の大きさがクラス生成次に指定した画像サイズと一致するか確認
        assert h == self.image_size and w == self.image_size

        # 高さ軸と幅軸をそれぞれパッチ数*パッチの大きさに分解して、
        # [バッチサイズ, チャネル数, パッチの行数, パッチの大きさ, パッチの列数, パッチの大きさ]
        # の形に変換
        x = x.view(
            bs,
            c,
            h // self.patch_size,
            self.patch_size,
            w // self.patch_size,
            self.patch_size,
        )

        # permute関数により
        # [バッチサイズ, チャネル数, パッチの行数, パッチの大きさ, パッチの列数, パッチの大きさ]
        # の形に変換
        x = x.permute(0, 2, 4, 1, 3, 5)

        # パッチを平坦化
        # permute関数適用後は、メモリ上のデータ配置の整合性の関係でview関数を使用的ないため、
        # reshape関数を使用
        x = x.reshape(bs, (h // self.patch_size) * (w // self.patch_size), -1)

        x = self.patch_embed(x)

        # クラス埋め込みをバッチサイズ分だけ用意
        class_token = self.class_token.expand(bs, -1, -1)

        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed

        # Transformerエンコーダー層を適用
        for layer in self.layers:
            x = layer(x)

        # クラス埋め込みをベースとした特徴量を抽出
        x = x[:, 0]
        x = self.norm(x)

        if return_embed:
            return x

        x = self.linear(x)
        return x

    def get_device(self) -> torch.device:
        """モデルパラメーターが保持されているデバイスを返す。"""
        return self.linear.weight.device

    def copy(self) -> Self:
        """モデルのコピーを返す。"""
        return copy.deepcopy(self)


class Config:
    """ハイパーパラメータとオプション"""

    def __init__(self) -> None:
        self.val_ratio = 0.2  # 検証に使う学習セット内のデータの割合
        self.patch_size = 4  # パッチサイズ
        self.dim_hidden = 512  # 隠れ層の次元
        self.num_heads = 8  # マルチヘッドアテンションのヘッド数
        self.dim_feedforward = 512  # Transformerエンコーダ層内のFNNにおける隠れ層の特徴量次元
        self.num_layers = 6  # Transformerエンコーダの層数
        self.num_epochs = 30  # 学習エポック数
        self.lr = 1e-2  # 学習率
        self.moving_avg = 20  # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32  # バッチサイズ
        self.num_workers = 2  # データローダに使うCPUプロセスの数
        # self.device = "cuda"  # 学習に使うデバイス
        self.device = "mps"  # 学習に使うデバイス
        self.num_samples = 200  # t-SNEでプロットするサンプル数


def train_eval():
    config = Config()

    # 入力データ正規化のために学習セットのデータを使って各チャネルの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=T.ToTensor()
    )
    channel_mean, channel_std = calculate_dataset_statistics_by_channels(dataset)

    # 画像の整形を行うクラスのインスタンスを用意
    transforms = T.Compose(
        (
            T.ToTensor(),
            T.Normalize(mean=channel_mean, std=channel_std),
        )
    )

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=True, download=True, transform=transforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir(), train=False, download=True, transform=transforms
    )

    # 学習・検証セットへ分割するためのインデックス集合の生成
    val_indices, train_indices = generate_subset_indices_pair(
        len(train_dataset), config.val_ratio
    )

    print(f"学習セットのサンプル数: {len(train_indices)}")
    print(f"検証セットのサンプル数: {len(val_indices)}")
    print(f"テストセットのサンプル数: {len(test_dataset)}")

    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_indices)

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
        sampler=val_indices,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # 目的関数の生成
    loss_func = F.cross_entropy

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float("inf")
    model_best = None

    # Vision Transformerモデルの生成
    model = VisionTransformer(
        len(train_dataset.classes),
        32,
        config.patch_size,
        config.dim_hidden,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
    )

    # モデルを指定デバイスに転送(デフォルトはGPU)
    model.to(config.device)

    # 最適化器の生成
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f"[エポック {epoch + 1}]")

            # 移動平均計算用
            losses = deque()
            accs = deque()
            for x, y in pbar:
                # データをモデルと同じデバイスに転送
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                # パラメータの勾配をリセット
                optimizer.zero_grad()

                # 順伝播
                y_pred = model(x)

                # 学習データに対する損失と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()

                # 誤差逆伝播
                loss.backward()

                # パラメータの更新
                optimizer.step()

                # 移動平均を計算して表示
                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix(
                    {
                        "loss": torch.Tensor(losses).mean().item(),
                        "accuracy": torch.Tensor(accs).mean().item(),
                    }
                )

        # 検証セットを使って精度評価
        val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
        print(f"検証: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

        # より良い検証結果が得られた場合、モデルを記録
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    # テスト
    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f"テスト: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")

    # t-SNEを使って特徴量の分布をプロット
    plot_t_sne(test_loader, model_best, config.num_samples)


if __name__ == "__main__":
    train_eval()
