from typing import Optional, Sequence, Union

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor


def constant_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    value: int = 0,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    if len(pad) != 4:
        raise RuntimeError(
            f"The length of pad {len(pad)} is not 4, which is not yet supported!"
        )

    if value != 0:
        raise RuntimeError(
            f"The padding value {value} is not 0, which is not yet supported!"
        )

    pre_padding = []
    post_padding = []
    for i in range(len(pad) - 1, -1, -2):
        post_padding.append(pad[i])
        pre_padding.append(pad[i - 1])

    # add padding layer
    pad_layer = ctx.net.add_padding_nd(
        input=input,
        pre_padding=tuple(pre_padding),
        post_padding=tuple(post_padding),
    )

    pad_layer.pre_padding_nd = tuple(pre_padding)
    pad_layer.post_padding_nd = tuple(post_padding)

    set_layer_name(pad_layer, target, name, source_ir)
    return pad_layer.get_output(0)
