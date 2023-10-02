from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR, get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def reshape(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    shape: List[int],
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    layer = ctx.net.add_shuffle(input)
    layer.reshape_dims = tuple(shape)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


# def flatten(
#     ctx: ConversionContext,
#     target: Union[Target, str],
#     source_ir: Optional[SourceIR],
#     name: str,
#     input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
#     start_dim: int,
#     end_dim: int,
# ) -> TRTTensor:
#     shape = input.shape
#     dim_size = len(shape)
#     start_dim = get_positive_dim(start_dim, dim_size)
#     end_dim = get_positive_dim(end_dim, dim_size)

#     if not isinstance(input, TRTTensor):
#         input = get_trt_tensor(ctx, input, f"{name}_input")

#     num_elements = 1
#     for i in range(start_dim, end_dim + 1):
#         num_elements *= shape[i]

#     new_shape = (
#         tuple(shape[:start_dim])
#         + (num_elements,)
#         + (tuple(shape[end_dim + 1 :]) if end_dim + 1 < dim_size else tuple())
#     )
#     layer = ctx.net.add_shuffle(input)
#     layer.reshape_dims = new_shape
#     set_layer_name(layer, target, name, source_ir)
#     return layer.get_output(0)
