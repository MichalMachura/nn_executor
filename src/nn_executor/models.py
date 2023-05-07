from typing import List, Tuple, Union
import torch
from torch import nn

from nn_executor.pruner_base import PrunerBase


DIFFERENTIATE_TENSOR = False


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        if DIFFERENTIATE_TENSOR:
            args = [a.clone() for a in args]

        if len(args) == 0:
            return None

        if len(args) == 1:
            return args[0]

        return args


class Absorber(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        return None



class Upsample(nn.Upsample):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode: str = "nearest",
        align_corners: bool = None,
    ) -> None:
        super().__init__(size, scale_factor, mode, align_corners)

    def extra_repr(self) -> str:
        s = f"size={self.size}, scale_factor={self.scale_factor}, " \
            + f"mode='{self.mode}', align_corners={self.align_corners}"
        return s


class CONSTANTS:

    BATCH_DIM_CAT_VAR_CONST = 1


class Constant(nn.Module):
    def __init__(self, t: torch.Tensor) -> None:
        super().__init__()
        self.t = torch.nn.Parameter(t, requires_grad=False)

    def __repr__(self, ) -> str:
        return f"Constant(t=torch.zeros({tuple(self.t.shape)}, dtype={self.t.dtype}))"

    def forward(self):
        return torch.cat([self.t] * CONSTANTS.BATCH_DIM_CAT_VAR_CONST, dim=0)

    def extra_repr(self) -> str:
        return "t=" + str(self.t)


class Variable(Constant):
    def __init__(self, t: torch.Tensor) -> None:
        super().__init__(t)
        self.t.requires_grad = True

    def __repr__(self) -> str:
        return f"Variable(t=torch.zeros({tuple(self.t.shape)}, dtype={self.t.dtype}))"


class Elementwise(nn.Module):
    def __init__(self, num, op) -> None:
        super().__init__()
        self.op = op
        self.num = num

    def extra_repr(self) -> str:
        return f"num={self.num}"

    def forward(self, *args):
        x = args[0]
        for a in args[1:]:
            x = self.op(x, a)

        # update num of args
        self.num = len(args)

        return x


class Add(Elementwise):
    def __init__(self, num: int = 1) -> None:
        super().__init__(num, torch.add)


class Sub(Elementwise):
    def __init__(self, num: int = 1) -> None:
        super().__init__(num, torch.sub)


class Mul(Elementwise):
    def __init__(self, num: int = 1) -> None:
        super().__init__(num, torch.mul)


class WithBatchNorm2d(nn.Module):
    def __init__(self, module: nn.Module, ch: int) -> None:
        super().__init__()
        self.module: nn.Module = module
        self.bn = nn.BatchNorm2d(ch)

    def __repr__(self) -> str:
        s = f"WithBatchNorm2d(module={self.module}, ch={self.bn.num_features})"
        return s

    def forward(self, *args):
        y = self.module(*args)
        y = self.bn(y)
        return y


class ModuleWithConstArgs(nn.Module):
    def __init__(
        self, module: nn.Module, modules: List[Tuple[torch.nn.Module, int]] = None
    ) -> None:
        super().__init__()
        self.module: nn.Module = module
        self.params_pos: List[Tuple[torch.nn.Module, int]] = []  # Constants or Variables modules

        if modules is not None:
            for mp in modules:
                self.add(mp[1], mp[0])

    def __repr__(self):
        s = f"{self.__class__.__name__}(module={self.module}, modules={self.params_pos})"
        return s

    def add(self, pos: int, module: nn.Module):
        with torch.no_grad():
            self.add_module(f"const_input_at_pos_{pos}", module)
            self.params_pos.append((module, pos))

    def forward(self, *args):
        args_iter = iter(args)
        joined_args = []

        params_pos = sorted(self.params_pos, key=lambda x: x[1])

        for param, pos in params_pos:
            # fill with input args
            while len(joined_args) != pos:
                input_ = next(args_iter)
                joined_args.append(input_)

            # add parameter to list of inputs
            joined_args.append(param())

        DST_LEN = len(args) + len(params_pos)
        while len(joined_args) != DST_LEN:
            input_ = next(args_iter)
            joined_args.append(input_)

        # forward by module
        y = self.module(*joined_args)

        return y


class ConvBnRelu(torch.nn.Module):
    def __init__(self,ch_in: int, ch_out: int, ks: int, padding: int = 0,
                 groups: int = 1, stride: int = 1, dilation: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, ks, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_inter, ks: int) -> None:
        super().__init__()
        self.conv1 = ConvBnRelu(ch_in, ch_inter, ks, padding=ks//2)
        self.conv2 = ConvBnRelu(ch_inter, ch_in, ks, padding=ks//2)
        self.adder = Add(2)

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.conv2(y)
        r = self.adder(x, y)
        return r


class ResBranch(nn.Module):
    def __init__(self, ch_in: int, ch_inter: int, ks: int = 1, num_blocks: int = 1) -> None:
        super().__init__()
        blocks = [ResBlock(ch_in, ch_inter, ks) for i in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        return x


class Cat(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim: int = dim
        self.input_shapes: List[Tuple] = []
        self.output_shape: Tuple = ()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        self.input_shapes = [t.shape for t in x]

        x = torch.cat(x, dim=self.dim)
        self.output_shape = x.shape

        return x


class ChannelsLogger(torch.nn.Module):
    def __init__(self, channels: List[int] = None) -> None:
        super().__init__()
        self.channels: List[int] = channels if channels is not None else []

    def extra_repr(self) -> str:
        return f"channels={self.channels}"

    def forward(self, *x: torch.Tensor):
        # update channels info
        self.channels = [t.shape[1] for t in x]
        return x


class OutputLayer(ChannelsLogger):
    pass


class InputLayer(ChannelsLogger):
    pass


def sigmoid(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    x = torch.exp(-a * x + b)
    return 1 / (1 + x)


class Pruner(PrunerBase):
    def __init__(
        self,
        ch: int,
        prunable: bool = False,
        activated: bool = True,
        threshold: float = 0.75,
        num_of_appearances: int = 1,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.threshold = threshold
        self.activated = activated
        self.prunable = prunable
        self.num_of_appearances = num_of_appearances
        self.pruner_weight = torch.nn.Parameter(torch.ones((1, self.ch, 1, 1),
                                                           dtype=torch.float32) * 1.0000008344650269,
                                                requires_grad=activated)
        self.init_ones()
        if activated:
            self.adjustment_mode()

    def extra_repr(self) -> str:
        s = f"ch={self.ch}, prunable={self.prunable}, activated={self.activated}, " \
            + "threshold={self.threshold}, num_of_appearances={self.num_of_appearances}"
        return s

    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.prunable and self.activated:
                mask = self.pruner_weight > self.threshold
            else:
                mask = torch.ones_like(self.pruner_weight, dtype=torch.bool)

            if self.activated:
                multipliers = self.forward(torch.ones(1, dtype=torch.float32))
            else:
                multipliers = torch.ones_like(
                    self.pruner_weight, dtype=torch.float32)

        return mask.flatten(), multipliers.flatten()

    def adjustment_mode(self):
        with torch.no_grad():
            device = self.pruner_weight.device
            self.pruner_weight[:] = (1 - 0.01 * torch.rand((1, self.ch, 1, 1),
                                                           dtype=torch.float32,
                                                           device=device))
        self.pruner_weight.requires_grad = True

    def init_ones(self):
        with torch.no_grad():
            self.pruner_weight[:] = 1.0000008344650269

    def forward(self, x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x

        s = sigmoid(self.pruner_weight, 100, 86)
        # x = x * (s * self.pruner_weight**2)
        m = s * self.pruner_weight
        with torch.no_grad():
            mask = (self.pruner_weight > self.threshold).to(torch.float32)
        # hard off
        m = mask * m
        # select channels by multiplication
        x = x * m

        return x


class AllOrNothingPruner(Pruner):

    def forward(self, x):
        if not self.activated:
            # cloning for differentiate input and output tensor
            return x.clone() if DIFFERENTIATE_TENSOR else x

        mean = self.pruner_weight.mean()
        s = sigmoid(mean, 100, 86)
        m = s * self.pruner_weight

        with torch.no_grad():
            mask = (mean > self.threshold).to(torch.float32)
            mask = torch.ones_like(self.pruner_weight) * mask

        # hard off
        m = mask * m
        # select channels by multiplication
        x = x * m

        return x

    def pruning_mask_and_multiplier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.prunable and self.activated:
                mask = self.pruner_weight.mean() > self.threshold
                mask = torch.ones_like(
                    self.pruner_weight, dtype=torch.bool) * mask
            else:
                mask = torch.ones_like(self.pruner_weight, dtype=torch.bool)

            if self.activated:
                multipliers = self.forward(torch.ones(1, dtype=torch.float32))
            else:
                multipliers = torch.ones_like(
                    self.pruner_weight, dtype=torch.float32)

        return mask.flatten(), multipliers.flatten()


class YOLO(torch.nn.Module):
    """
    You Only Look Once:
    Assume N as num of anchors and C as num of classes.\n
    Validity - N channels are transformed with sigmoid function.\n
    Classification - C*N channels are transformed with 'cls_fcn' function.\n
    X object center coord - N channels are transformed with sigmoid function and there is added column idx.\n
    Y object center coord - N channels are transformed with sigmoid function and there is added row idx.\n
    X and Y coordinates are then rescaled to original image scale.\n
    Width - N channels are transformed with exponential function and multiplied by anchors width.\n
    Height - N channels are transformed with exponential function and multiplied by anchors height.\n

    Inputs: tensor to transform and network image (to obtain it's shape).\n
    Return tensor after YOLO transformation applied:\n
    Output channels are grouped by type (instead of anchors) in the following order:\n
    [V, CLS, X, Y, W, H]\n
    """

    def __init__(
        self, anchors: List[Union[List[int], int]], cls_fcn: str = "sigmoid"
    ) -> None:
        """
        _summary_

        :param anchors: list of shape (N,2) or (N*2,).
            Sequence of anchors sizes (width, height).
        :param cls_fcn: str, one of ['sigmoid', 'softmax', 'softmin']
        """
        super().__init__()
        anchors = anchors.reshape((1, -1, 2, 1)).to(torch.float32)
        self._anchors = torch.nn.Parameter(anchors, requires_grad=False)
        self.cls_fcn = cls_fcn
        self.grid_X = None
        self.grid_Y = None
        self.create_grid()

    @property
    def anchors(self):
        return self._anchors

    def extra_repr(self):
        s = f"anchors={self.anchors.reshape(-1,2).detach().tolist()}, cls_fcn='{self.cls_fcn}'"
        return s

    def create_grid(self, HW=(10, 20)):
        x_indices = torch.arange(0, HW[1], dtype=torch.long)
        self.grid_X = x_indices.reshape(1, 1, 1, -1).repeat(1, 1, HW[0], 1)

        y_indices = torch.arange(0, HW[0], dtype=torch.long)
        self.grid_Y = y_indices.reshape(1, 1, -1, 1).repeat(1, 1, 1, HW[1])

    def forward(self, x: torch.Tensor, network_input: torch.Tensor):
        device = x.device
        num_of_anchors = self.anchors.numel() // 2
        noa = num_of_anchors

        out_W, out_H = x.shape[-2:][::-1]
        in_W, in_H = network_input.shape[-2:][::-1]
        scale_W, scale_H = in_W / out_W, in_H / out_H

        # update grid
        if self.grid_X.shape[-1] != out_W or self.grid_X.shape[-2] != out_H:
            with torch.no_grad():
                self.create_grid((out_H, out_W))

        self.grid_X = self.grid_X.to(device)
        self.grid_Y = self.grid_Y.to(device)

        V = torch.sigmoid(x[:, :noa, :, :])

        CLS = x[:, noa: -4 * noa, :, :]
        if self.cls_fcn == "sigmoid":
            CLS = torch.sigmoid(CLS)

        elif self.cls_fcn == "softmax":
            sh = CLS.shape
            CLS = CLS.reshape(sh[0], -1, noa, *sh[2:])
            CLS = torch.softmax(CLS, dim=1).reshape(sh)

        elif self.cls_fcn == "softmin":
            sh = CLS.shape
            CLS = CLS.reshape(sh[0], -1, noa, *sh[2:])
            CLS = torch.softmax(-CLS, dim=1).reshape(sh)

        X = (torch.sigmoid(x[:, -4 * noa: -3 * noa, :, :]
                           ) + self.grid_X) * scale_W
        Y = (torch.sigmoid(x[:, -3 * noa: -2 * noa, :, :]
                           ) + self.grid_Y) * scale_H

        W = torch.exp(x[:, -2 * noa: -noa, :, :]) * self.anchors[:, :, :1, :]
        H = torch.exp(x[:, -noa:, :, :]) * self.anchors[:, :, 1:, :]

        OUT = torch.cat([V, CLS, X, Y, W, H], dim=1)

        return OUT


class YOLOAnchorMul(YOLO):
    """
    Anchors are multiplied by scalers as non negative values.

    Module contains additional parameter 'anchors_mul'.

    This parameter is first transformed exponentially.

    The result then is multiplied with original anchors.
    """

    def __init__(self, anchors: torch.Tensor) -> None:
        super().__init__(anchors)
        self.anchors_mul = torch.nn.Parameter(
            torch.rand_like(anchors) * 0.1 - 0.05, requires_grad=True
        )

    @property
    def anchors(self):
        mul = torch.exp(self.anchors_mul)
        return self._anchors * mul


class Parallel(nn.Module):
    def __init__(self, merger: nn.Module, *branches: Union[List[nn.Module], nn.Module]) -> None:
        super().__init__()
        self.merger:nn.Module = merger
        joined = []

        for b in branches:
            if not isinstance(b, list):
                b = [b]
            joined.extend(b)

        self.branches: List[nn.Module] = joined

        for i, b in enumerate(self.branches):
            self.add_module("Branch_" + str(i), b)

    def forward(self, x):
        results = [B(x) for B in self.branches]

        return self.merger(*results)


# if __name__ == "__main__":

#     # find weight which gives multiplier equal to 1.0
#     def fcn(p):
#         return p*sigmoid(p,100,86)
#     w = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
#     lr = 0.0001
#     E = torch.inf
#     for i in range(100):
#         # loss computing
#         v = fcn(w)
#         e = torch.abs(1-v)
#         # get grad
#         e.backward()
#         g = w.grad
#         # update
#         with torch.no_grad():
#             w -= g*lr
#         lr *= 1.2 if e < E else 1/2
#         E = e.item()
#         print(f"W: {w.item()} value: {v.item()} E: {E} g: {g.item()} lr: {lr}")
#         w.grad *= 0
