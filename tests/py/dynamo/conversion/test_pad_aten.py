import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestConstantPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 3, 4, 2), (1, 2, 3, 4), 0),
            ((3, 3, 4, 2, 1), (1, 2, 3, 4), 0),
            ((3, 3, 4, 2, 1, 2), (1, 2, 3, 4), 0),
        ]
    )
    def test_constant_pad(self, shape, pad, value):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.constant_pad_nd.default(input, pad, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


# class TestPadConverter(DispatchTestCase):
#     @parameterized.expand(
#         [
#             ((3, 3, 4, 2), (1, 1, 2, 2), "constant", 0),
#             # (),
#             # (),
#             # (),
#             # (),
#         ]
#     )
#     def test_pad(self, shape, pad, mode, value):
#         class TestModule(torch.nn.Module):
#             def forward(self, input):
#                 return torch.ops.aten.pad.default(input, pad, mode, value)

#         input = [torch.randn(shape)]
#         self.run_test(
#             TestModule(),
#             input,
#         )


if __name__ == "__main__":
    run_tests()
