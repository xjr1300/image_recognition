import numpy as np
from scipy import signal


def gaussian_kernel(width: int, height: int, sigma: float) -> np.ndarray:
    """ガウシアンカーネルを生成する。

    Args:
        width (int): カーネルの幅
        height (int): カーネルの高さ
        sigma (float): カーネルの値を決定するガウス分布の標準偏差
    Returns:
        np.ndarray: ガウシアンカーネル
    """
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("width and height must be odd numbers.")
    kernel = np.empty((height, width))
    for y in range(-(height // 2), height // 2 + 1):
        for x in range(-(width // 2), width // 2 + 1):
            # ガウス分布を計算してカーネルに格納
            h = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            kernel[y + height // 2, x + width // 2] = h
    # カーネルの総和が1になるように正規化
    kernel /= kernel.sum()
    return kernel


def edge_kernels():
    """エッジを検出するカーネルを生成する。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 水平方向、垂直方向及びラプラシアンフィルタを実現するカーネルを格納したタプル
    """
    # カーネル用の変数を準備
    kernel_h = np.zeros((3, 3))
    kernel_v = np.zeros((3, 3))
    kernel_lap = np.zeros((3, 3))
    # 水平方向の一次微分のカーネル
    kernel_h[1, 1] = -1
    kernel_h[1, 2] = 1
    # 垂直方向の一次微分のカーネル
    kernel_v[1, 1] = -1
    kernel_v[2, 1] = 1
    # ラプラシアンフィルタのカーネル
    kernel_lap[0, 1] = 1
    kernel_lap[1, 0] = 1
    kernel_lap[1, 1] = -4
    kernel_lap[1, 2] = 1
    kernel_lap[2, 1] = 1
    return kernel_h, kernel_v, kernel_lap
