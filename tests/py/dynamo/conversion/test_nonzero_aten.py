import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestNonZeroConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,),),
            ((3, 4),),
            ((3, 4, 5),),
        ]
    )
    def test_nonzero_int(self, shape):
        class NonZero(nn.Module):
            def forward(self, x):
                return torch.ops.aten.nonzero.default(x)

        inputs = [torch.randint(0, 3, shape, dtype=torch.int32)]
        self.run_test(
            NonZero(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,),),
            ((3, 4),),
            ((3, 4, 5),),
        ]
    )
    def test_nonzero_float(self, shape):
        class NonZero(nn.Module):
            def forward(self, x):
                return torch.ops.aten.nonzero.default(x)

        inputs = [torch.randn(shape)]
        self.run_test(
            NonZero(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
