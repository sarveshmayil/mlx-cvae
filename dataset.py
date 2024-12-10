import gzip
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import request

import numpy as np
from mlx.data import buffer_from_vector
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file


def load_mnist(save_dir: str = None) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
    """Downloads the MNIST dataset and saves it as a safetensors file.

    Args:
        save_dir (`str`): The directory where the dataset will be saved. Defaults to ".cache/mnist".

    Returns:
        A dictionary containing the training and test sets of the MNIST dataset.
    """
    save_dir = Path(save_dir or ".cache/mnist")
    save_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train": {
            "images": "train-images-idx3-ubyte.gz",
            "labels": "train-labels-idx1-ubyte.gz",
        },
        "test": {
            "images": "t10k-images-idx3-ubyte.gz",
            "labels": "t10k-labels-idx1-ubyte.gz",
        },
    }

    tempdir = TemporaryDirectory()
    tempdir_path = Path(tempdir.name)
    mnist = {
        "train": defaultdict(NDArray),
        "test": defaultdict(NDArray),
    }
    for split, files in filenames.items():
        save_path = save_dir / f"{split}.safetensors"
        if not save_path.exists():
            print(f"Downloading MNIST {split} files...")
            for name in files.values():
                url = f"https://raw.githubusercontent.com/fgnt/mnist/master/{name}"
                download_path = tempdir_path / name
                request.urlretrieve(url, download_path)

            for data_type, name in files.items():
                with gzip.open(tempdir_path / name, "rb") as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1) if data_type == "images" else np.frombuffer(f.read(), np.uint8, offset=8)
                mnist[split][data_type] = data.copy()  # Need to make a copy because data is created by a memory view, which is read-only

            save_file(mnist[split], save_path)
        else:
            mnist[split] = load_file(save_path)

    return mnist["train"], mnist["test"]


def mnist(batch_size: int, image_shape: tuple[int, int]):
    """Returns the MNIST dataset.

    Args:
        batch_size (`int`): The batch size.
        image_shape (`tuple`): The shape of the images (H, W).

    Returns:
        A tuple containing the training and test dataloaders.
    """
    train, test = load_mnist()

    train_dataset = (
        buffer_from_vector([{"image": im, "label": lb} for im, lb in zip(train["images"], train["labels"], strict=False)])
        .shuffle()
        .to_stream()
        .image_resize("image", h=image_shape[0], w=image_shape[1])
        .key_transform("image", lambda x: x.astype("float32") / 255.0)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test_dataset = (
        buffer_from_vector([{"image": im, "label": lb} for im, lb in zip(test["images"], test["labels"], strict=False)])
        .to_stream()
        .image_resize("image", h=image_shape[0], w=image_shape[1])
        .key_transform("image", lambda x: x.astype("float32") / 255.0)
        .batch(batch_size)
    )

    return train_dataset, test_dataset
