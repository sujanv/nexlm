import torch
import pytest

from training.amp import MixedPrecisionManager


def test_mixed_precision_manager_cpu_and_types():
    device = torch.device("cpu")
    mgr_fp16 = MixedPrecisionManager(device, precision="fp16")
    mgr_bf16 = MixedPrecisionManager(device, precision="bf16")
    mgr_fp32 = MixedPrecisionManager(device, precision="fp32")

    assert mgr_fp16.dtype == torch.float16
    assert mgr_bf16.dtype == torch.bfloat16
    assert mgr_fp32.dtype == torch.float32


def test_mixed_precision_training_step():
    device = torch.device("cpu")
    mgr = MixedPrecisionManager(device, precision="fp32")

    linear = torch.nn.Linear(8, 2)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    x = torch.randn(4, 8)
    y = torch.randn(4, 2)

    with mgr.autocast_context():
        pred = linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)

    mgr.scale_and_backward(loss)
    mgr.step(optimizer, model=linear, grad_clip=1.0)

    assert linear.weight.grad is not None
