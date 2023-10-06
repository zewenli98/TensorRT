import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestReshapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((-1,),),
            ((20,),),
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    def test_reshape(self, target_shape):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.reshape.default(x, target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(
            Reshape(),
            inputs,
        )

    @parameterized.expand(
        [
            ((-1,),),
            ((-1, 10),),
            ((-1, 5),),
            ((2, 2, -1),),
        ]
    )
    def test_reshape_with_dynamic_shape(self, target_shape):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.reshape.default(x, target_shape)

        input_specs = [
            Input(
                shape=(-1, 2, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 5), (10, 2, 5), (10, 2, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Reshape(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((-1,),),
            ((20,),),
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    def test_view(self, target_shape):
        class View(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.view.default(x, target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(
            View(),
            inputs,
        )

    @parameterized.expand(
        [
            ((-1,),),
            ((-1, 10),),
            ((-1, 5),),
            ((2, 2, -1),),
        ]
    )
    def test_view_with_dynamic_shape(self, target_shape):
        class View(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.view.default(x, target_shape)

        input_specs = [
            Input(
                shape=(-1, 2, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 5), (10, 2, 5), (10, 2, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            View(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
