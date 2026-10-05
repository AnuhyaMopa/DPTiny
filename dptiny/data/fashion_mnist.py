"""Fashion-MNIST dataset loader."""
import gzip
import os
import struct
import urllib.request
from typing import Optional, Tuple
import numpy as np
from dptiny.backend import xp
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")
_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "zalandoresearch/fashion-mnist/master/data/fashion/"
)
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
def _download_file(url: str, destination: str) -> None:
    """Download a file atomically so interrupted downloads are not cached."""
    if os.path.exists(destination):
        return

    temp_file = destination + ".tmp"

    try:
        # Remove an old temporary file if one exists.
        if os.path.exists(temp_file):
            os.remove(temp_file)

        urllib.request.urlretrieve(url, temp_file)

        # Atomic replacement: destination is only created after
        # the complete download succeeds.
        os.replace(temp_file, destination)

    except BaseException:
        # Never leave a half-written cache file.
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise
def _read_idx_gz(
    filename: str,
    expected_dimensions: int,
) -> np.ndarray:
    """Read and validate a gzipped uint8 IDX file."""

    with gzip.open(filename, "rb") as f:
        raw = f.read()

    # IDX header:
    # bytes 0-1: zero
    # byte 2: data type
    # byte 3: number of dimensions
    if len(raw) < 4:
        raise ValueError("Invalid IDX file: header is too short.")

    zero1, zero2, data_type, ndim = struct.unpack(">BBBB", raw[:4])

    # IDX uint8 type code is 0x08.
    if zero1 != 0 or zero2 != 0 or data_type != 0x08:
        raise ValueError(
            "Invalid IDX file: expected a uint8 IDX file."
        )

    if ndim != expected_dimensions:
        raise ValueError(
            f"Invalid IDX file: expected {expected_dimensions} dimensions, "
            f"found {ndim}."
        )

    # Each dimension size is stored as a 4-byte big-endian integer.
    header_size = 4 + 4 * ndim

    if len(raw) < header_size:
        raise ValueError("Invalid IDX file: incomplete dimension header.")

    dimensions = struct.unpack(
        ">" + "I" * ndim,
        raw[4:header_size],
    )

    expected_data_size = int(np.prod(dimensions))
    actual_data_size = len(raw) - header_size

    if actual_data_size != expected_data_size:
        raise ValueError(
            "Invalid IDX file: data size does not match the dimensions "
            "stored in the header."
        )

    data = np.frombuffer(
        raw,
        dtype=np.uint8,
        offset=header_size,
        count=expected_data_size,
    )

    return data.reshape(dimensions)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """Load the Fashion-MNIST dataset.

    Args:
        normalize: If True, scale pixel values to [0, 1].
        flatten: If True, return images as (N, 784) vectors.
        data_home: Directory used to cache the downloaded dataset.

    Returns:
        (X_train, X_test, y_train, y_test)
    """

    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Download all four files if they are not already cached.
    local_files = {}

    for key, filename in _FILES.items():
        destination = os.path.join(cache_dir, filename)
        url = _BASE_URL + filename

        _download_file(url, destination)
        local_files[key] = destination

    # Parse the IDX files using our own NumPy parser.
    train_images = _read_idx_gz(
        local_files["train_images"],
        expected_dimensions=3,
    )

    train_labels = _read_idx_gz(
        local_files["train_labels"],
        expected_dimensions=1,
    )

    test_images = _read_idx_gz(
        local_files["test_images"],
        expected_dimensions=3,
    )

    test_labels = _read_idx_gz(
        local_files["test_labels"],
        expected_dimensions=1,
    )

    # Make sure the image and label counts agree.
    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("Training images and labels have different sizes.")

    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("Test images and labels have different sizes.")

    # Convert to the same dtypes as get_mnist().
    X_train = train_images.astype(np.float32)
    X_test = test_images.astype(np.float32)

    y_train = train_labels.astype(np.int32)
    y_test = test_labels.astype(np.int32)

    # Match get_mnist() normalization behavior.
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # Match get_mnist() shape behavior.
    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    # Convert NumPy arrays to DPTiny's active array type.
    X_train = xp.asarray(X_train, dtype=xp.float32)
    X_test = xp.asarray(X_test, dtype=xp.float32)
    y_train = xp.asarray(y_train, dtype=xp.int32)
    y_test = xp.asarray(y_test, dtype=xp.int32)

    return X_train, X_test, y_train, y_test
