from typing import List, Tuple, Type, Union
from warnings import warn
import torch
from torch import nn
from nn_executor import models, utils
from nn_executor.types import FORWARD_TYPE, BACKWARD_TYPE


class Modifier:
    def __init__(self) -> None:
        pass

    def clone(self, in_module: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "You must implement this method in any subclasses!")

    def forward(self,
                in_module: nn.Module,
                in_mask_mul_bias: List[FORWARD_TYPE],
                # (out_mask, out_mul) - out_mul is not None if node cannot fuse it or proagate backward
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        # return module, [(non_zero_mask, None, Bias),] -- bias must be fused with multiplier
        # if all module' outs are bias -> module = None,
        # return module, [(non_zero_mask, Mul, None),] -- output multiplier cannot be fused with this module
        # return module,[(non_zero_mask,None,None)] -- output multiplier fused
        # return None, [(zeroed_mask,None,None)]
        return None

    def backward(self,
                 in_module: nn.Module,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],  # new module
                            List[BACKWARD_TYPE],  # backward for inputs
                            # backward for output (mask, mul)
                            List[BACKWARD_TYPE],
                            ]:                   # if out_mul is propagated backward
        # or fused, then mul should be None
        return None


class Conv2dModifier(Modifier):

    def clone(self, in_module: nn.Conv2d) -> nn.Module:
        m = nn.Conv2d(in_module.in_channels,
                      in_module.out_channels,
                      in_module.kernel_size,
                      in_module.stride,
                      in_module.padding,
                      in_module.dilation,
                      in_module.groups,
                      in_module.bias is not None,
                      in_module.padding_mode)
        m.load_state_dict(in_module.state_dict())
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
            raise RuntimeError(
                f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

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

        if BIAS is not None:
            return None, [(out_mask, None, BIAS)]

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


class BatchNorm2dModifier(Modifier):

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
            raise RuntimeError(
                f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

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

                    m.weight[:] = (in_module.weight*mul)[in_mask]
                    m.bias[:] = (in_module.bias*mul)[in_mask]
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
                BIAS = B - W*M/torch.sqrt(V+EPS)

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


class VariableModifier(Modifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: models.Variable) -> nn.Module:
        with torch.no_grad():
            m = models.Variable(in_module.t)

        return m

    def forward(self,
                in_module: models.Variable,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        out_mask, out_mul = out_mask_mul[0]
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        with torch.no_grad():
            BIAS = in_module.t.flatten()
            return None, [(out_mask, None, BIAS),]

    def backward(self,
                 in_module: models.Variable,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, multiplier = out_mask_multipliers[0]

        with torch.no_grad():
            m = models.Variable((in_module.t*multiplier))

        return m, [], [(mask, None),]


class ConstantModifier(Modifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: models.Constant) -> nn.Module:
        with torch.no_grad():
            m = models.Constant(in_module.t)
        return m

    def forward(self,
                in_module: models.Constant,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        out_mask, out_mul = out_mask_mul[0]
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        with torch.no_grad():
            # out_mul should be None
            BIAS = in_module.t.flatten()
            return None, [(out_mask, None, BIAS),]

    def backward(self,
                 in_module: models.Constant,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, multiplier = out_mask_multipliers[0]

        with torch.no_grad():
            m = models.Constant((in_module.t*multiplier))

        return m, [], [(mask, None),]


# NOT FOR DIRECT USE
class PassActivationModifier(Modifier):
    def __init__(self, act: Type[torch.nn.Module] = models.Identity) -> None:
        super().__init__()
        self.act = act

    def clone(self, in_module: nn.Module) -> nn.Module:
        return self.act()

    def process_bias(self,
                     in_module: nn.Module,
                     BIAS: torch.Tensor):
        with torch.no_grad():
            m = self.clone(in_module).eval()
            BIAS = m.forward(BIAS).flatten()
            return BIAS

    def forward(self,
                in_module: nn.Module,
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
            raise RuntimeError(
                f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

        if ch_in > 0:
            # input is const
            if in_bias is not None:
                with torch.no_grad():
                    BIAS = self.process_bias(in_module, in_bias)
                    return None, [(in_mask, None, BIAS),]
            # input is variable
            else:
                m = self.clone(in_module)
                #
                if in_mul is not None and out_mul is not None:
                    mul = in_mul * out_mul
                elif in_mul is not None:
                    mul = in_mul
                elif out_mul is not None:
                    mul = out_mul
                else:
                    mul = None

                return m, [(in_mask, mul, None),]
        # no input -> no output
        else:
            return None, [(in_mask, None, None)]

    def backward(self,
                 in_module: nn.Module,
                 in_mask_multipliers: List[Tuple[torch.Tensor,
                                                 torch.Tensor]]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, mul = in_mask_multipliers[0]
        m = self.clone(in_module)
        # assumption -> multiplier can be propagated to previous layer
        # without self and module changes !!!
        return m, [(mask, mul),], [(mask, None)]


class UpsampleModifier(PassActivationModifier):
    def __init__(self) -> None:
        super().__init__(models.Upsample)

    def clone(self, in_module: models.Upsample) -> nn.Module:
        m = models.Upsample(
            in_module.size,
            in_module.scale_factor,
            in_module.mode,
            in_module.align_corners
        )
        return m

    def process_bias(self,
                     in_module: nn.Module,
                     BIAS: torch.Tensor):
        return BIAS.clone()


class ReLUModifier(PassActivationModifier):
    def __init__(self) -> None:
        super().__init__(torch.nn.ReLU)

    def clone(self, in_module: nn.ReLU) -> nn.Module:
        m = nn.ReLU(in_module.inplace)
        return m


class LeakyReLUModifier(PassActivationModifier):
    def __init__(self) -> None:
        super().__init__(torch.nn.LeakyReLU)

    def clone(self, in_module: nn.LeakyReLU) -> nn.Module:
        m = nn.LeakyReLU(in_module.negative_slope, in_module.inplace)
        return m


class IdentityModifier(PassActivationModifier):
    def __init__(self) -> None:
        super().__init__(models.Identity)


class MaxPool2dModifier(PassActivationModifier):
    def __init__(self) -> None:
        super().__init__()

    def process_bias(self, in_module: nn.Module, BIAS: torch.Tensor):
        return BIAS.detach().clone()

    def clone(self, in_module: nn.MaxPool2d) -> nn.Module:
        m = nn.MaxPool2d(in_module.kernel_size,
                         in_module.stride,
                         in_module.padding,
                         in_module.dilation,
                         in_module.return_indices,
                         in_module.ceil_mode)
        return m


# TODO
class ElementwiseModifier(Modifier):
    def __init__(self, ew_mod_cls) -> None:
        super().__init__()
        self.ew_mod_cls = ew_mod_cls

    def clone(self, in_module: models.Elementwise) -> nn.Module:
        m = self.ew_mod_cls(in_module.num)
        return m

    def forward(self, in_module: nn.Module, in_mask_mul_bias: List[FORWARD_TYPE], out_mask_mul: List[BACKWARD_TYPE]) -> Tuple[Union[nn.Module, None], List[FORWARD_TYPE]]:
        raise NotImplementedError()

    def backward(self,
                 in_module: models.Elementwise,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, mul = out_mask_multipliers[0]

        m = self.clone(in_module)
        # # for mul it muls should be splitted for each input or passed to only one input
        return m, [(mask, mul) for i in range(in_module.num)], [(mask, None),]


class AddModifier(ElementwiseModifier):
    def __init__(self, use_bias: bool = False) -> None:
        super().__init__(models.Add)
        self.__use_bias = use_bias

    def forward(self,
                in_module: models.Add,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        # raise NotImplementedError()
        out_mask, out_mul = out_mask_mul[0]
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        biases = []
        masks = []
        for i, (in_mask, in_mul, in_bias) in enumerate(in_mask_mul_bias):

            if in_mul is not None:
                # TODO injection of premultipliers for some of inputs
                raise RuntimeError(
                    "AddModifier cannot apply multiplier for input")

            ch_in = in_mask.sum().item()
            if ch_in == 0 and in_bias is not None:
                raise RuntimeError(
                    f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

            if ch_in > 0:
                # constant input
                if in_bias is not None:
                    biases.append(in_bias)
                    continue
                # variable input
                masks.append(in_mask)

        BIAS = sum(biases) if biases else None

        # only bias if available
        if len(masks) == 0:
            return None, [(out_mask, None, BIAS),]

        mask = utils.between_all(torch.logical_and, masks)
        mask_or = utils.between_all(torch.logical_or, masks)
        ch = mask.sum().item()
        xor = mask != mask_or
        ch_xor = xor.sum().item()

        if ch_xor > 0:
            raise RuntimeError("Input masks are different")

        if ch != ch_out:
            warn(f"Inputs masks channels:{ch} is different from output mask:{ch_out}.")

        if ch < ch_out:
            warn(f"Outputs masks channels:{ch_out} is reduced to:{ch} channels.")

        # adder with input for bias
        m = models.Add(len(masks)+int(BIAS is not None))

        if BIAS is not None and self.__use_bias:
            b = models.Variable(BIAS[mask].reshape(1, -1, 1, 1))
            m = models.ModuleWithConstArgs(m)
            m.add(0, b)

        return m, [(mask, None, None),]


# TODO -> needs different flow
# class SubModifier(ElementwiseModifier):
#     def __init__(self) -> None:
#         super().__init__(models.Sub)


class MulModifier(ElementwiseModifier):
    def __init__(self, use_root=True) -> None:
        super().__init__(models.Mul)
        self.use_root = use_root

    def forward(self,
                in_module: models.Mul,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        # raise NotImplementedError()
        out_mask, out_mul = out_mask_mul[0]
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        muls = []
        masks = [out_mask]
        for i, (in_mask, in_mul, in_bias) in enumerate(in_mask_mul_bias):
            masks.append(in_mask)

            ch_in = in_mask.sum().item()
            if ch_in == 0 and in_bias is not None:
                raise RuntimeError(
                    f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

            # constant input
            if in_bias is not None:
                muls.append(in_bias)
                continue
            # multiplied input
            if in_mul is not None:
                muls.append(in_mul)

        MUL = utils.between_all(torch.mul, muls) if len(muls) > 0 else None
        # there is min one element -- out_mask
        MASK = utils.between_all(torch.logical_and, masks)
        CH = MASK.sum().item()

        # only bias if available
        if CH == 0:
            return None, [(out_mask, None, None),]

        num_of_muls = len(in_mask_mul_bias) + int(MUL is not None)
        m = models.Mul(num_of_muls)

        if MUL is not None:
            b = models.Variable(MUL.reshape(1, -1, 1, 1))
            m = models.ModuleWithConstArgs(m)
            m.add(0, b)

        return m, [(MASK, None, None),]

    def backward(self,
                 in_module: models.Mul,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, mul = out_mask_multipliers[0]

        m = self.clone(in_module)

        if self.use_root:
            exponent = torch.tensor(1 / in_module.num)
            root = torch.pow(mul, exponent)
            return m, [(mask, root) for i in range(in_module.num)], [(mask, None),]

        # tensors of ones are backpropagated to inputs, mul is stored for applying after (MultiWithConst)
        return m, [(mask, torch.ones_like(mul)) for i in range(in_module.num)], [(mask, mul),]


class CatModifier(Modifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: models.Cat) -> nn.Module:
        m = models.Cat(in_module.dim)
        m.input_shapes = in_module.input_shapes
        m.output_shape = in_module.output_shape

        return m

    def forward(self,
                in_module: models.Cat,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        # raise NotImplementedError()
        out_mask, out_mul = out_mask_mul[0]
        ch_out = out_mask.sum().item()

        # output of this module is not needed
        if ch_out == 0:
            return None, [(out_mask, None, None),]

        dim = in_module.dim
        muls = []
        biases = []
        muls_cntr = 0
        pos_available = 0
        out_masks = []
        valid_shapes = []
        valid_biases = []

        for i, (in_mask, in_mul, in_bias) in enumerate(in_mask_mul_bias):
            out_masks.append(in_mask)
            # filled out muls
            if in_mul is not None:
                muls.append(in_mul)
                muls_cntr += 1
            else:
                muls.append(torch.ones_like(in_mask,
                                            dtype=torch.float32))
            # filled out biases
            if in_bias is not None:
                biases.append(in_bias)
            else:
                biases.append(torch.zeros_like(in_mask,
                                               dtype=torch.float32))

            ch_in = in_mask.sum().item()
            if ch_in == 0 and in_bias is not None:
                raise RuntimeError(
                    f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

            if ch_in > 0:
                # update shape of this input
                sh = list(in_module.input_shapes[i])
                sh[dim] = ch_in
                valid_shapes.append(sh)
                # input is const
                if in_bias is not None:
                    valid_biases.append((pos_available, in_bias[in_mask], sh))
                # next available pos
                pos_available += 1

        MASK = torch.cat(out_masks, dim=0)
        MUL = torch.cat(muls, dim=0)
        BIAS = torch.cat(biases, dim=0)

        # no input biases
        if len(valid_biases) == 0:
            # no muls
            if muls_cntr == 0:
                MUL = None
            m = models.Cat(dim)
            m.input_shapes = valid_shapes
            out_shape = list(in_module.output_shape)
            out_shape[dim] = MASK.sum().item()
            m.output_shape = out_shape
            return m, [(MASK, MUL, None),]

        # full input is bias
        if len(valid_biases) == len(valid_shapes):
            return None, [(MASK, None, BIAS),]

        # some inputs are biases -> Cat + Variable
        if len(valid_biases) < len(valid_shapes):
            # with this module can integrate bias,
            # so out mul can be available
            if muls_cntr == 0:
                MUL = None

            # base cat for variable inputs and constants
            m = models.Cat(dim)
            m.input_shapes = valid_shapes
            out_shape = list(in_module.output_shape)
            out_shape[dim] = MASK.sum().item()
            m.output_shape = out_shape

            # m = models.ModuleWithConstArgs(m)
            # for pos, bias, sh in valid_biases:
            #     # bias shape adjust to dim
            #     b_sh = [1 for s in sh]
            #     b_sh[1] = -1 # channel dim
            #     # dims of bias map -- shape of origina map that is replace by bias
            #     bias_map_sh = list(sh)
            #     bias_map_sh[dim] = 1

            #     b = torch.tile(bias.reshape(b_sh),dims=bias_map_sh)
            #     # pack in module
            #     b = models.Variable(b)
            #     m.add(pos,b)

            return m, [(MASK, MUL, None),]

    def backward(self,
                 in_module: models.Cat,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        out_mask, out_mul = out_mask_multipliers[0]
        # split mask on output of module to it's inputs
        mask_mul = []
        idx = 0

        # split onto input signals
        for shape in in_module.input_shapes:
            ch_of_input = idx + shape[in_module.dim]
            # get part which comes from current input
            mask = out_mask[idx:ch_of_input]
            mul = out_mul[idx:ch_of_input]
            # add to list
            mask_mul.append((mask, mul))
            # update idx
            idx = ch_of_input

        m = self.clone(in_module)

        return m, mask_mul, [(out_mask, None)]


class OutputLayerModifier(Modifier):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: models.OutputLayer) -> nn.Module:
        return models.OutputLayer(in_module.channels.copy())

    def forward(self,
                in_module: models.OutputLayer,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        channels = []
        for in_mask, in_mul, in_bias in in_mask_mul_bias:
            ch = in_mask.sum().item()
            if ch == 0 or in_bias is not None:
                warn("Removed one of outputs.")
                continue
            if ch < in_mask.flatten().shape[0]:
                raise RuntimeError(f"Output layer should not be pruned!!!")
            if in_mul is not None:
                raise RuntimeError(f"Output layer does not support in_mul")
            if in_bias is not None:
                raise RuntimeError(f"Output layer does not support in_bias")
        m = models.OutputLayer(channels)

        # return self.clone(in_module), []
        return m, []

    def backward(self,
                 in_module: models.OutputLayer,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask_muls = [(torch.ones(ch, dtype=torch.bool),
                      torch.ones(ch, dtype=torch.float32)) for ch in in_module.channels]

        return self.clone(in_module), mask_muls, []


class InputLayerModifier(Modifier):

    def __init__(self) -> None:
        super().__init__()

    def clone(self, in_module: models.InputLayer) -> nn.Module:
        return models.InputLayer(in_module.channels)

    def forward(self,
                in_module: models.InputLayer,
                in_mask_mul_bias: List[FORWARD_TYPE],
                out_mask_mul: List[BACKWARD_TYPE],
                ) -> Tuple[Union[nn.Module, None],
                           List[FORWARD_TYPE]]:
        # raise NotImplementedError()
        out_mask, out_mul = out_mask_mul[0]

        # ch_in = [m[0].sum().item()
        #          for m in in_mask_mul_bias if m[0].sum().item() > 0]

        # m = models.InputLayer(ch_in)
        m = self.clone(in_module)

        return m, [(mask, mul, None) for mask, mul in out_mask_mul]

    def backward(self,
                 in_module: models.InputLayer,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        new_out_mask_multipliers = [(mask, None) for mask, mul in out_mask_multipliers]
        return self.clone(in_module), [], new_out_mask_multipliers


class PrunerModifier(Modifier):
    def __init__(self, replace_with_identity=False) -> None:
        super().__init__()
        self.replace_with_identity = replace_with_identity

    def clone(self, in_module: models.Pruner) -> models.Pruner:
        m = models.Pruner(in_module.ch, in_module.prunable,
                          in_module.activated, in_module.threshold)
        with torch.no_grad():
            m.pruner_weight[:] = in_module.pruner_weight[:]
        return m

    def forward(self,
                in_module: models.Pruner,
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
            raise RuntimeError(
                f"Input mask channel with {ch_in} non zero elements and bias cannot be used together!")

        if ch_in > 0:
            # input is const
            if in_bias is not None:
                return None, [(in_mask, None, in_bias),]
            # input is variable
            if not self.replace_with_identity:
                m = self.clone(in_module)
                with torch.no_grad():
                    # set multipliers to 1
                    m.init_ones()
            else:
                m = models.Identity()
            return m, [(in_mask, in_mul, None),]
        # no input
        # output is bias of this module
        return None, [(in_mask, None, None)]

    def backward(self,
                 in_module: models.Pruner,
                 out_mask_multipliers: List[BACKWARD_TYPE]
                 ) -> Tuple[Union[nn.Module, None],
                            List[BACKWARD_TYPE],
                            List[BACKWARD_TYPE],
                            ]:
        mask, mul = out_mask_multipliers[0]
        c_mask, c_mul = in_module.pruning_mask_and_multiplier()

        mask = torch.logical_and(mask, c_mask)

        if not self.replace_with_identity:
            m = self.clone(in_module)
            with torch.no_grad():
                # set multipliers to 1
                m.init_ones()
        else:
            m = models.Identity()

        in_mul = mul*c_mul
        in_mask = mask

        return m, [(in_mask, in_mul)], [(in_mask, None),]
