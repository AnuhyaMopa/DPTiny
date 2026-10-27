"""Autograd ``Function`` implementations for DPTiny."""

from typing import Tuple, Union

from dptiny import utils
from dptiny.backend import xp
from dptiny.core import Config, Function, Variable


class Add(Function):
    def forward(self, x0, x1):
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self._x0_shape != self._x1_shape:
            gx0 = utils.sum_to(gx0, self._x0_shape)
            gx1 = utils.sum_to(gx1, self._x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1.data
        gx1 = gy * x0.data
        if self._x0_shape != self._x1_shape:
            gx0 = utils.sum_to(gx0, self._x0_shape)
            gx1 = utils.sum_to(gx1, self._x1_shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self._x0_shape != self._x1_shape:
            gx0 = utils.sum_to(gx0, self._x0_shape)
            gx1 = utils.sum_to(gx1, self._x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self._x0_shape = x0.shape
        self._x1_shape = x1.shape
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1.data
        gx1 = gy * (-x0.data / x1.data**2)
        if self._x0_shape != self._x1_shape:
            gx0 = utils.sum_to(gx0, self._x0_shape)
            gx1 = utils.sum_to(gx1, self._x1_shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, exponent: float):
        super().__init__()
        self._exponent = exponent

    def forward(self, x):
        return x**self._exponent

    def backward(self, gy):
        x = self.inputs[0]
        return gy * self._exponent * (x.data ** (self._exponent - 1))


class Exp(Function):
    def forward(self, x):
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y.data


class Log(Function):
    def forward(self, x):
        return xp.log(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy / x.data


class Tanh(Function):
    def forward(self, x):
        return xp.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y.data * y.data)


class Sigmoid(Function):
    def forward(self, x):
        self._positive_mask = x >= 0
        y = xp.where(
            self._positive_mask,
            1 / (1 + xp.exp(-x)),
            xp.exp(x) / (1 + xp.exp(x)),
        )
        self._y = y
        return y

    def backward(self, gy):
        return gy * self._y * (1 - self._y)


class ReLU(Function):
    def forward(self, x):
        return xp.maximum(x, 0)

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        return gy * mask


class MatMul(Function):
    def forward(self, x, W):
        self.x_shape = x.shape
        self.W_shape = W.shape
        return x @ W

    def backward(self, gy):
        x, W = self.inputs
        gx = gy @ W.data.T
        gW = x.data.T @ gy

        if gx.shape != self.x_shape:
            gx = gx.reshape(self.x_shape)
        if gW.shape != self.W_shape:
            gW = gW.reshape(self.W_shape)

        return gx, gW


class Linear(Function):
    """Fused affine transform ``y = x @ W + b``."""

    def forward(self, x, W, b=None):
        self.has_bias = b is not None
        y = x @ W
        if self.has_bias:
            y = y + b
        return y

    def backward(self, gy):
        x, W = self.inputs[0], self.inputs[1]
        gx = gy @ W.data.T
        gW = x.data.T @ gy
        if gx.shape != x.data.shape:
            gx = gx.reshape(x.data.shape)
        if gW.shape != W.data.shape:
            gW = gW.reshape(W.data.shape)
        if self.has_bias:
            gb = utils.sum_to(gy, self.inputs[2].data.shape)
            return gx, gW, gb
        return gx, gW


class Reshape(Function):
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape
        self.x_shape: Tuple[int, ...] = ()

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return gy.reshape(self.x_shape)


class Transpose(Function):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        self.x_shape = x.shape
        return x.transpose(self.axes)

    def backward(self, gy):
        if self.axes is None:
            return gy.transpose()
        inv = xp.argsort(xp.array(self.axes))
        return gy.transpose(tuple(int(a) for a in inv))


class Sum(Function):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims
        )
        return xp.broadcast_to(gy, self.x_shape)


class Mean(Function):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        if self.axis is None:
            self.divisor = x.size
        else:
            axes = (
                (self.axis,)
                if isinstance(self.axis, int)
                else tuple(self.axis)
            )
            self.divisor = 1
            for a in axes:
                self.divisor *= x.shape[a]
        return x.sum(axis=self.axis, keepdims=self.keepdims) / self.divisor

    def backward(self, gy):
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims
        )
        return xp.broadcast_to(gy / self.divisor, self.x_shape)


class Max(Function):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.max(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        y = self.outputs[0]()
        gy_b = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims
        )
        y_b = utils.reshape_sum_backward(
            y.data, self.x_shape, self.axis, self.keepdims
        )
        gy_b = xp.broadcast_to(gy_b, self.x_shape)
        y_b = xp.broadcast_to(y_b, self.x_shape)
        x = self.inputs[0]
        mask = x.data == y_b
        count = mask.sum(
            axis=self.axis, keepdims=True
        )
        return gy_b * mask / count


class GetItem(Function):
    def __init__(self, slices):
        super().__init__()
        self.slices = slices

    def forward(self, x):
        self.x_shape = x.shape
        return x[self.slices]

    def backward(self, gy):
        return GetItemGrad(self.slices, self.x_shape)(gy).data


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        super().__init__()
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return ggx[self.slices]


class BroadcastTo(Function):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return xp.broadcast_to(x, self.shape)

    def backward(self, gy):
        return utils.sum_to(gy, self.x_shape)


class SumTo(Function):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy):
        return xp.broadcast_to(gy, self.x_shape)


class Softmax(Function):
    def forward(self, x):
        y = x - x.max(axis=1, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=1, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y.data * gy
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= y.data * sumdx
        return gx


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        self.t = t
        self.x_shape = x.shape

        batch_size = x.shape[0]
        if self.t.size == self.t.shape[0]:  # labels are not one-hot
            self.t_oh = xp.eye(x.shape[1], dtype=x.dtype)[self.t]
        else:
            self.t_oh = self.t

        # Numerically stable softmax
        x_max = x.max(axis=1, keepdims=True)
        x_shifted = x - x_max
        exp_x = xp.exp(x_shifted)
        sum_exp_x = exp_x.sum(axis=1, keepdims=True)
        self.y = exp_x / sum_exp_x

        # Numerically stable log-softmax
        log_y = x_shifted - xp.log(sum_exp_x)
        batch_log_y = xp.sum(self.t_oh * log_y, axis=1)
        loss = -xp.sum(batch_log_y) / batch_size
        return loss

    def backward(self, gy):
        batch_size = self.x_shape[0]
        dx = (self.y - self.t_oh) * gy / batch_size
        return dx


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        diff = x0 - x1
        self.diff = diff
        return (diff**2).sum() / diff.size

    def backward(self, gy):
        diff = self.diff
        gy = xp.broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / diff.size)
        gx1 = -gx0
        if self.x0_shape != self.x1_shape:
            gx0 = utils.sum_to(gx0, self.x0_shape)
            gx1 = utils.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Dropout(Function):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x):
        if Config.train and self.p > 0:
            self.mask = (xp.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
            return x * self.mask
        self.mask = None
        return x.copy() if hasattr(x, "copy") else x

    def backward(self, gy):
        if self.mask is not None:
            return gy * self.mask
        return gy


class BatchNorm(Function):
    """Batch normalization over the batch (and spatial) axes.

    ``running_mean``/``running_var`` are raw arrays updated in place when
    ``Config.train`` is ``True``.
    """

    def __init__(self, momentum: float = 0.9, eps: float = 2e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, gamma, beta, running_mean, running_var):
        self.train = Config.train
        if x.ndim == 2:
            self.axes = (0,)
            bshape = (1, -1)
        elif x.ndim == 4:
            self.axes = (0, 2, 3)
            bshape = (1, -1, 1, 1)
        else:
            raise ValueError("batch_norm supports 2D or 4D inputs")

        gamma_b = gamma.reshape(bshape)
        beta_b = beta.reshape(bshape)

        if self.train:
            m = 1
            for a in self.axes:
                m *= x.shape[a]
            # Use biased variance for normalization, like the standard BN.
            mean = x.mean(axis=self.axes, keepdims=True)
            var = x.var(axis=self.axes, keepdims=True)
            running_mean[...] = (
                self.momentum * running_mean
                + (1 - self.momentum) * mean.reshape(running_mean.shape)
            )
            running_var[...] = (
                self.momentum * running_var
                + (1 - self.momentum) * var.reshape(running_var.shape)
            )
            self.m = m
        else:
            mean = running_mean.reshape(bshape)
            var = running_var.reshape(bshape)

        self.std = xp.sqrt(var + self.eps)
        self.xhat = (x - mean) / self.std
        self.gamma_b = gamma_b
        return self.xhat * gamma_b + beta_b

    def backward(self, gy):
        gamma = self.inputs[1]
        gbeta = utils.sum_to(gy, (gamma.data.shape[0],))
        ggamma = utils.sum_to(gy * self.xhat, (gamma.data.shape[0],))
        if self.train:
            gxhat = gy * self.gamma_b
            gx = (self.gamma_b / self.std) / self.m * (
                self.m * gy
                - gy.sum(axis=self.axes, keepdims=True)
                - self.xhat * (gxhat * self.xhat).sum(
                    axis=self.axes, keepdims=True
                )
            )
        else:
            gx = gy * self.gamma_b / self.std
        return gx, ggamma, gbeta


class Conv2d(Function):
    """2-D cross-correlation (convolution) with learnable kernel.

    Input shape: ``(N, C, H, W)``
    Weight shape: ``(OC, C, KH, KW)``
    Output shape: ``(N, OC, OH, OW)``

    Supports stride, padding, and dilation.
    """

    def __init__(
        self,
        stride: Union[int, Tuple[int, int]] = 1,
        pad: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):
        super().__init__()

        self.stride = utils.pair(stride)
        self.pad = utils.pair(pad)
        self.dilation = utils.pair(dilation)

        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation

        if SH < 1 or SW < 1:
            raise ValueError("stride must be >= 1")

        if PH < 0 or PW < 0:
            raise ValueError("pad must be >= 0")

        if DH < 1 or DW < 1:
            raise ValueError("dilation must be >= 1")

    def forward(self, x, W):
        N, C, H, W_in = x.shape
        OC, C2, KH, KW = W.shape

        if C != C2:
            raise ValueError(
                f"Input channels {C} do not match weight channels {C2}"
            )

        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation

        # Effective kernel size after dilation.
        KH_eff = DH * (KH - 1) + 1
        KW_eff = DW * (KW - 1) + 1

        # The effective kernel must fit inside the padded input.
        padded_H = H + 2 * PH
        padded_W = W_in + 2 * PW

        if KH_eff > padded_H or KW_eff > padded_W:
            raise ValueError(
                "effective kernel does not fit inside padded input: "
                f"effective kernel=({KH_eff}, {KW_eff}), "
                f"padded input=({padded_H}, {padded_W})"
            )

        OH = (padded_H - KH_eff) // SH + 1
        OW = (padded_W - KW_eff) // SW + 1

        x_padded = (
            xp.pad(
                x,
                ((0, 0), (0, 0), (PH, PH), (PW, PW)),
                mode="constant",
            )
            if PH > 0 or PW > 0
            else x
        )

        # Create windows using the effective kernel size.
        patches = xp.lib.stride_tricks.sliding_window_view(
            x_padded,
            (KH_eff, KW_eff),
            axis=(2, 3),
        )

        # Apply stride to output positions and dilation inside each window.
        patches = patches[
            :,
            :,
            ::SH,
            ::SW,
            ::DH,
            ::DW,
        ]

        # (N, C, OH, OW, KH, KW)
        # -> (N*OH*OW, C*KH*KW)
        col = patches.transpose(
            0, 2, 3, 1, 4, 5
        ).reshape(
            N * OH * OW,
            C * KH * KW,
        )

        W_col = W.reshape(OC, -1)

        out = col @ W_col.T
        out = out.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)

        self.input_shape = x.shape
        self.col_shape = (N, OH, OW, C, KH, KW)
        self._col = col

        return out

    def backward(self, gy):
        x, W = self.inputs

        x_data = x.data
        W_data = W.data

        N, C, H, W_in = x_data.shape
        OC, _, KH, KW = W_data.shape

        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation

        N, OC, OH, OW = gy.shape

        W_col = W_data.reshape(OC, -1)

        gy_col = gy.transpose(
            0, 2, 3, 1
        ).reshape(
            N * OH * OW,
            OC,
        )

        # Gradient with respect to W.
        gW_col = gy_col.T @ self._col
        gW = gW_col.reshape(OC, C, KH, KW)

        # Gradient with respect to the input columns.
        gcol = gy_col @ W_col

        gcol = gcol.reshape(
            N,
            OH,
            OW,
            C,
            KH,
            KW,
        ).transpose(
            0,
            3,
            1,
            2,
            4,
            5,
        )

        # Allocate gradient for padded input.
        gx_padded = xp.zeros(
            (
                N,
                C,
                H + 2 * PH,
                W_in + 2 * PW,
            )
            if PH > 0 or PW > 0
            else (
                N,
                C,
                H,
                W_in,
            ),
            dtype=x_data.dtype,
        )

        # Build indices for scatter-add.
        n_idx = xp.arange(N).reshape(
            N, 1, 1, 1, 1, 1
        )

        c_idx = xp.arange(C).reshape(
            1, C, 1, 1, 1, 1
        )

        oh_idx = xp.arange(OH).reshape(
            1, 1, OH, 1, 1, 1
        )

        ow_idx = xp.arange(OW).reshape(
            1, 1, 1, OW, 1, 1
        )

        kh_idx = xp.arange(KH).reshape(
            1, 1, 1, 1, KH, 1
        )

        kw_idx = xp.arange(KW).reshape(
            1, 1, 1, 1, 1, KW
        )

        # Dilation is applied when scattering gradients.
        h_idx = oh_idx * SH + kh_idx * DH
        w_idx = ow_idx * SW + kw_idx * DW

        xp.add.at(
            gx_padded,
            (
                n_idx,
                c_idx,
                h_idx,
                w_idx,
            ),
            gcol,
        )

        # Remove padding from gradient.
        gx = (
            gx_padded[
                :,
                :,
                PH : H + PH,
                PW : W_in + PW,
            ]
            if PH > 0 or PW > 0
            else gx_padded
        )

        return gx, gW


class MaxPool2d(Function):
    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = utils.pair(kernel_size)
        self.stride = (
            self.kernel_size if stride is None else utils.pair(stride)
        )
        self.pad = utils.pair(pad)

    def forward(self, x):
        col = utils.im2col_array(
            x, self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        return col.max(axis=2)

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        KH, KW = self.kernel_size
        dcol = xp.zeros((N, C, KH * KW, OH, OW), dtype=gy.dtype)
        xp.put_along_axis(
            dcol, self.indexes[:, :, None, :, :], gy[:, :, None, :, :], axis=2
        )
        dcol = dcol.reshape(N, C, KH, KW, OH, OW)
        x_shape = self.inputs[0].data.shape
        return utils.col2im_array(
            dcol, x_shape, self.kernel_size, self.stride, self.pad,
            to_matrix=False,
        )


class AvgPool2d(Function):
    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = utils.pair(kernel_size)
        self.stride = (
            self.kernel_size if stride is None else utils.pair(stride)
        )
        self.pad = utils.pair(pad)

    def forward(self, x):
        col = utils.im2col_array(
            x, self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        N, C, KH, KW, OH, OW = col.shape
        self.khkw = KH * KW
        col = col.reshape(N, C, KH * KW, OH, OW)
        return col.mean(axis=2)

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        KH, KW = self.kernel_size
        dcol = xp.broadcast_to(
            gy[:, :, None, :, :] / self.khkw, (N, C, KH * KW, OH, OW)
        ).copy()
        dcol = dcol.reshape(N, C, KH, KW, OH, OW)
        x_shape = self.inputs[0].data.shape
        return utils.col2im_array(
            dcol, x_shape, self.kernel_size, self.stride, self.pad,
            to_matrix=False,
        )


# ---------------------------------------------------------------------------
# Thin wrapper functions
# ---------------------------------------------------------------------------


def add(x0, x1):
    return Add()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    return Sub()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


def rsub(x0, x1):
    return sub(x1, x0)


def rdiv(x0, x1):
    return div(x1, x0)


def exp(x):
    return Exp()(x)


def log(x):
    return Log()(x)


def tanh(x):
    return Tanh()(x)


def sigmoid(x):
    return Sigmoid()(x)


def relu(x):
    return ReLU()(x)


def softmax(x):
    return Softmax()(x)


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def matmul(x, W):
    return MatMul()(x, W)


def linear(x, W, b=None):
    if b is None:
        return Linear()(x, W)
    return Linear()(x, W, b)


def pow(x, exponent):
    return Pow(exponent)(x)


def reshape(x, shape):
    if x.shape == shape:
        return x if isinstance(x, Variable) else Variable(x)
    return Reshape(shape)(x)


def transpose(x, axes=None):
    return Transpose(axes)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims)(x)


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def broadcast_to(x, shape):
    if getattr(x, "shape", None) == shape:
        return x if isinstance(x, Variable) else Variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    if getattr(x, "shape", None) == shape:
        return x if isinstance(x, Variable) else Variable(x)
    return SumTo(shape)(x)


def get_item(x, slices):
    return GetItem(slices)(x)


def dropout(x, p=0.5):
    return Dropout(p)(x)


def batch_norm(
    x, gamma, beta, running_mean, running_var, momentum=0.9, eps=2e-5
):
    return BatchNorm(momentum, eps)(x, gamma, beta, running_mean, running_var)


def conv2d(x, W, stride=1, pad=0, dilation=1):
    return Conv2d(
        stride=stride,
        pad=pad,
        dilation=dilation,
    )(x, W)


def max_pool2d(x, kernel_size, stride=None, pad=0):
    return MaxPool2d(kernel_size, stride, pad)(x)


def avg_pool2d(x, kernel_size, stride=None, pad=0):
    return AvgPool2d(kernel_size, stride, pad)(x)


def flatten(x):
    """Reshape ``(N, ...)`` input to ``(N, -1)``."""
    return reshape(x, (x.shape[0], -1))


def accuracy(y, t):
    """Classification accuracy between predictions ``y`` and labels ``t``."""
    pred = y.data.argmax(axis=1)
    if t.ndim != 1:
        t = t.argmax(axis=1)
    acc = (pred == t).mean()
    return Variable(xp.array(acc))
