import os


def package_dir() -> str:
    """パッケージディレクトリを返す。

    Returns:
        str: パッケージディレクトリのパス。
    """
    return os.path.dirname(os.path.abspath(__file__))


def resource_dir() -> str:
    """リソースディレクトリを返す。

    Returns:
        str: リソースディレクトリのパス。
    """
    return os.path.join(package_dir(), "resources")


def image_resource_dir() -> str:
    """イメージリソースディレクトリを返す。

    Returns:
        str: イメージリソースディレクトリのパス。
    """
    return os.path.join(resource_dir(), "images")


def cifar_dir() -> str:
    """CIFAR-10データセットのディレクトリを返す。

    Returns:
        str: CIFAR-10データセットのディレクトリのパス。
    """
    path = os.path.join(resource_dir(), "cifar10")
    os.makedirs(path, exist_ok=True)
    return path
