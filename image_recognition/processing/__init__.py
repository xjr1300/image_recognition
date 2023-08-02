import numpy as np
from PIL import Image


def convolution(image: Image.Image, kernel: np.ndarray, x: int, y: int) -> float:
    """入力画像の指定したピクセルに対して、畳み込み演算を実行する。

    Args:
        image (np.ndarray): 入力画像
        kernel (np.ndarray): 畳み込み演算に適用するカーネル([height, width]])
        x (int): 畳み込み演算するピクセルのx座標
        y (int): 畳み込み演算するピクセルのy座標
    Returns:
        畳み込み演算したピクセル値
    """
    if len(image.getbands()) != 1:
        raise ValueError("image must be a grayscale image.")
    # 画像サイズを取得
    width, height = image.size
    kernel_height, kernel_width = kernel.shape[:2]
    # 畳み込み演算
    value = 0
    # カーネルの高さが5の場合、-2, -1, 0, 1, 2をrangeで生成
    for y_kernel in range(-(kernel_height // 2), kernel_height // 2 + 1):
        # カーネルの幅が5の場合、-2, -1, 0, 1, 2をrangeで生成
        for x_kernel in range(-(kernel_width // 2), kernel_width // 2 + 1):
            # カーネルが画像からはみ出る場合は、端の座標を設定
            # 例えば、画像の幅が100ピクセルで、カーネルの幅が5ピクセルの場合、
            # xが1でx_kernelが-2の場合次のようになる。
            # max(min(1 + (-2), 100 - 1), 0)
            # = max(min(-1, 99), 0)
            # = max(-1, 0)
            # = 0
            # xが98でx_kernelが2の場合次のようになる。
            # max(min(98 + 2), 100 - 1), 0)
            # = max(min(100, 99), 0)
            # = max(99, 0)
            # = 99
            x_index = max(min(x + x_kernel, width - 1), 0)
            y_index = max(min(y + y_kernel, height - 1), 0)
            h = kernel[y_kernel + kernel_height // 2, x_kernel + kernel_width // 2]
            value += h * image.getpixel((x_index, y_index))
    return value


def apply_convolution(image: Image, kernel: np.ndarray) -> Image:
    """画像を畳み込み演算する。

    Args:
        image (Image): フィルタを適用する画像
        kernel (np.ndarray): フィルタのカーネル([height, width])
    Returns:
        Image: フィルタを適用した画像
    """
    # 入力画像のサイズを取得
    width, height = image.size
    # フィルタを適用した結果を保存する単バンド画像を生成
    filtered_image = Image.new(mode="L", size=(width, height))

    for y in range(height):
        for x in range(width):
            value = convolution(image, kernel, x, y)
            filtered_image.putpixel((x, y), int(value))
    return filtered_image
