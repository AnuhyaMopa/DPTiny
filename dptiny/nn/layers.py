"""Standard neural-network layers for DPTiny."""

from typing import Optional, Sequence, Tuple, Union

from dptiny import functions as F
from dptiny import utils
from dptiny.backend import xp
from dptiny.core import Parameter
from dptiny.nn.module import Module

ShapeLike = Union[int, Sequence[int]]


def _init_weight(
    shape: Tuple[int, ...],
    init_type: str,
    fan_in: int,
    dtype=xp.float32,
) -> "xp.ndarray":
    """Create a weight array using a standard initializer."""
    if init_type == "he":
        scale = xp.sqrt(2.0 / fan_in)
    elif init_type == "xavier":
        scale = xp.sqrt(1.0 / fan_in)
    elif init_type == "xavier_uniform":
        limit = xp.sqrt(6.0 / (fan_in + shape[-1]))
        return xp.random.uniform(-limit, limit, shape).astype(dtype)
    elif init_type == "normal":
        scale = 0.01
    else:
        raise ValueError(
            f"unknown init_type {init_type!r}; expected one of "
            "'he', 'xavier', 'xavier_uniform', 'normal'"
        )
    return xp.random.randn(*shape).astype(dtype) * scale


class Linear(Module):
    """Fully connected (affine) layer.

    Input shape: ``(N, in_size)``
    Output shape: ``(N, out_size)``
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        nobias: bool = False,
        init_type: str = "he",
        dtype=xp.float32,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        W_data = _init_weight(
            (in_size, out_size), init_type, in_size, dtype
        )
        self.W = Parameter(W_data)
        self.b: Optional[Parameter] = (
            None if nobias else Parameter(xp.zeros(out_size, dtype=dtype))
        )

    def forward(self, x):
        return F.linear(x, self.W, self.b)


class Conv2d(Module):
    """2-D convolution layer.

    Input shape: ``(N, C, H, W)``
    Output shape: ``(N, out_channels, OH, OW)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ShapeLike,
        stride: Union[int, Tuple[int, int]] = 1,
        pad: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        nobias: bool = False,
        init_type: str = "he",
        dtype=xp.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = utils.pair(kernel_size)
        self.stride = stride
        self.pad = pad
        self.dilation = dilation

        KH, KW = self.kernel_size
        fan_in = in_channels * KH * KW

        W_data = _init_weight(
            (out_channels, in_channels, KH, KW), init_type, fan_in, dtype
        )
        self.W = Parameter(W_data)
        self.b: Optional[Parameter] = (
            None if nobias else Parameter(xp.zeros(out_channels, dtype=dtype))
        )

    def forward(self, x):
        y = F.conv2d(
            x,
            self.W,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
        )
        if self.b is not None:
            y = y + F.reshape(self.b, (1, self.out_channels, 1, 1))
        return y


class MaxPool2d(Module):
    """2-D max pooling layer."""

    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.pad)


class AvgPool2d(Module):
    """2-D average pooling layer."""

    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.pad)


class Flatten(Module):
    """Flatten ``(N, ...)`` inputs to ``(N, -1)``."""

    def forward(self, x):
        return F.flatten(x)


class Dropout(Module):
    """Dropout layer active only in training mode."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p)


class BatchNorm(Module):
    """Batch normalization over ``(N, C)`` or ``(N, C, H, W)`` inputs."""

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        eps: float = 2e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter(xp.ones(num_features, dtype=xp.float32))
        self.beta = Parameter(xp.zeros(num_features, dtype=xp.float32))
        self.register_buffer(
            "running_mean", xp.zeros(num_features, dtype=xp.float32)
        )
        self.register_buffer(
            "running_var", xp.ones(num_features, dtype=xp.float32)
        )

    def forward(self, x):
        return F.batch_norm(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            momentum=self.momentum,
            eps=self.eps,
        )


class ReLU(Module):
    """ReLU activation as a layer."""

    def forward(self, x):
        return F.relu(x)


class Sigmoid(Module):
    """Sigmoid activation as a layer."""

    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    """Tanh activation as a layer."""

    def forward(self, x):
        return F.tanh(x)
