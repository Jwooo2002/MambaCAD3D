import pytest
import torch

import main_files.model as implementation
from main_files.model import MemModule, SS3D


def test_grouped_direction_parameters_reach_their_own_state_scan(monkeypatch):
    # Each direction has distinct dt/B/C sentinels. The old global split
    # routed group-0 state values into group-1's delta projection.
    layer = SS3D(d_model=1, d_state=1, dt_rank=1, ssm_ratio=1)
    with torch.no_grad():
        layer.x_proj.weight.zero_()
        layer.x_proj.bias.copy_(torch.tensor([1., 2., 3., 10., 20., 30.]))
        layer.dt_proj.weight.fill_(1.)
        layer.dt_proj.bias.zero_()
    captured = {}

    def scan(**kwargs):
        captured.update(kwargs)
        return kwargs["u"]

    monkeypatch.setattr(implementation, "selective_scan_fn", scan)
    layer(torch.ones(1, 1, 2, 2, 2))
    torch.testing.assert_close(captured["B"][0, :, 0, 0], torch.tensor([2., 20.]))
    torch.testing.assert_close(captured["C"][0, :, 0, 0], torch.tensor([3., 30.]))
    torch.testing.assert_close(captured["delta"][0, :, 0], torch.nn.functional.softplus(torch.tensor([1., 10.])))


def test_memory_suppresses_below_threshold_attention_and_normalizes_survivors():
    memory = MemModule(mem_dim=3, fea_dim=1, shrink_thres=.2)
    with torch.no_grad():
        memory.memory.copy_(torch.tensor([[2.], [0.], [-2.]]))
    value = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
    output, attention = memory(value)
    torch.testing.assert_close(attention, torch.tensor([[[1., 0., 0.]]]))
    assert output.item() == pytest.approx(2.)
    output.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_all_suppressed_memory_is_explicit_zero_without_silent_softmax_fallback():
    memory = MemModule(mem_dim=3, fea_dim=1, shrink_thres=.5)
    _, attention = memory(torch.zeros(1, 1, 1, 1, 1))
    assert attention.sum() == 0
