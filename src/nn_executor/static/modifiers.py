from typing import Tuple, List, Union
from torch import nn
import torch
from nn_executor import modifiers
from nn_executor import types
from nn_executor.types import FORWARD_TYPE, BACKWARD_TYPE


class StaticModifier(modifiers.Modifier):

    def _norm(self,
              x: torch.Tensor,
              offset: torch.Tensor = None,
              dim=1,
              keepdim=True) -> torch.Tensor:
        if offset is not None:
            weights = weights - offset.reshape(-1, 1)
        x = x / (x.square().sum(dim, keepdim=keepdim).sqrt())
        return x

    def analyze_weights(self,
                        weights: torch.Tensor,
                        bias: torch.Tensor = None,
                        threshold=0.7
                        ):
        original_shape = weights.shape
        weights = weights.reshape(original_shape[0], -1)
        weights = self._norm(weights, offset=bias)

        # correlation matrix
        corr = torch.matmul(weights, weights.transpose(0, 1))

        # highly correlated filters mask
        corr_mask = corr.abs() > threshold

        # print(corr.shape)

        # over diagonal matrix
        row_idx, col_idx = torch.meshgrid(torch.arange(0, corr.shape[0]), torch.arange(0, corr.shape[0]),)
        analysis_mask = col_idx > row_idx

        # over diagonal part of mask
        filters_correlation = corr_mask * analysis_mask.to(weights.device)

        channels_mask = filters_correlation.sum(0) == 0

        return channels_mask


class Conv2dStaticModifier(StaticModifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, module: nn.Conv2d) -> nn.Module:
        m = nn.Conv2d(module.in_channels,
                      module.out_channels,
                      module.kernel_size,
                      module.stride,
                      module.padding,
                      module.dilation,
                      module.groups,
                      module.bias is not None,
                      module.padding_mode)
        m.load_state_dict(module.state_dict())

        return m

    def forward(self,
                in_module: nn.Conv2d,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        in_mask, in_mul, in_bias = in_mask_mul_bias[0]
        out_mask, out_mul = out_mask_mul[0]
        ch_in = in_mask.sum().item()
        ch_out = out_mask.sum().item()

        with torch.no_grad():
            W = in_module.weight.detach().clone()
            BIAS = in_module.bias.detach().clone() if in_module.bias is not None else None

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        if ch_in == 0 and in_bias is not None:
            raise RuntimeError(f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

        # input is not fully reduced
        if ch_in > 0:
            # input is const
            if in_bias is not None:
                bias = W * in_bias.reshape(1, -1, 1, 1)
                bias = bias.sum(dim=(1, 2, 3)).flatten()
                # add bias of this conv
                if BIAS is not None:
                    bias += BIAS

                return None, [(out_mask, None, bias)]

            # input has some variability
            else:
                m = nn.Conv2d(ch_in,
                              ch_out,
                              in_module.kernel_size,
                              in_module.stride,
                              in_module.padding,
                              in_module.dilation,
                              in_module.groups,
                              in_module.bias is not None,
                              in_module.padding_mode)

                with torch.no_grad():
                    # fuse with in mul
                    if in_mul is not None:
                        W = W * in_mul.reshape(1, -1, 1, 1)
                    # apply masks
                    W = W[out_mask, :, :, :][:, in_mask, :, :]
                    m.weight[:] = W

                    if BIAS is not None:
                        m.bias[:] = BIAS[out_mask]

                return m, [(out_mask, out_mul, None)]

        # no input
        else:
            if BIAS is not None:
                return None, [(out_mask, None, BIAS)]
            else:
                # module output is none -> zeroing mask
                return None, [(torch.zeros_like(out_mask), None, None)]

    def backward(self,
                 in_module: nn.Conv2d,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, multiplier = out_mask_multipliers[0]
        ch_out = mask.sum().item()

        m = self.clone(in_module)
        with torch.no_grad():
            m.weight[:] = in_module.weight * multiplier.reshape(-1, 1, 1, 1)
            if in_module.bias is not None:
                m.bias[:] = (in_module.bias * multiplier)

        if ch_out > 0:
            in_mask = torch.ones(in_module.in_channels, dtype=torch.bool)
            in_mul = torch.ones(in_module.in_channels, dtype=torch.float32)
        else:
            in_mask = torch.zeros(in_module.in_channels, dtype=torch.bool)
            in_mul = torch.ones(in_module.in_channels, dtype=torch.float32)

        out_mask = mask

        return m, [(in_mask, in_mul),], [(out_mask, None),]


class BatchNorm2dStaticModifier(StaticModifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: nn.BatchNorm2d) -> nn.Module:
        m = nn.BatchNorm2d(in_module.num_features,
                           in_module.eps,
                           in_module.momentum,
                           in_module.affine,
                           in_module.track_running_stats
                           ).eval()
        m.load_state_dict(in_module.state_dict())
        return m

    def forward(self,
                in_module: nn.BatchNorm2d,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        in_mask, in_mul, in_bias = in_mask_mul_bias[0]
        out_mask, out_mul = out_mask_mul[0]
        ch_in = in_mask.sum().item()
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        if ch_in == 0 and in_bias is not None:
            raise RuntimeError(f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

        if ch_in > 0:
            # input is const
            if in_bias is not None:
                with torch.no_grad():
                    m = self.clone(in_module).eval()
                    BIAS = m(in_bias.reshape(1, -1, 1, 1)).flatten()
                    return None, [(in_mask, out_mul, BIAS),]
            # input is variable
            else:
                m = nn.BatchNorm2d(ch_in,
                                   in_module.eps,
                                   in_module.momentum,
                                   in_module.affine,
                                   in_module.track_running_stats,
                                   ).eval()

                with torch.no_grad():
                    mul = 1
                    if in_mul is not None:
                        mul = in_mul

                    m.weight[:] = (in_module.weight * mul)[in_mask]
                    m.bias[:] = (in_module.bias * mul)[in_mask]
                    m.running_mean[:] = in_module.running_mean[in_mask]
                    m.running_var[:] = in_module.running_var[in_mask]
                    m.num_batches_tracked *= 0
                    m.num_batches_tracked += in_module.num_batches_tracked

                return m, [(in_mask, None, None),]

        # no input
        else:
            # output is bias of this module
            with torch.no_grad():
                W = in_module.weight
                B = in_module.bias
                M = in_module.running_mean
                V = in_module.running_var
                EPS = in_module.eps
                BIAS = B - W * M / torch.sqrt(V + EPS)

                return None, [(out_mask, None, BIAS),]

    def backward(self,
                 in_module: nn.BatchNorm2d,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, multiplier = out_mask_multipliers[0]
        m = self.clone(in_module)

        with torch.no_grad():
            m.weight[:] *= multiplier
            m.bias[:] *= multiplier

        return m, [(mask, torch.ones_like(mask))], [(mask, None)],
