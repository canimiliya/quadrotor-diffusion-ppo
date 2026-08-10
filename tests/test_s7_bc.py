import torch

from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC


def test_matched_bc_shape_scale_and_support():
    model = MatchedSequenceBC()
    output = model(torch.randn(7, 34))
    assert output.shape == (7, 16, 3)
    assert sum(p.numel() for p in model.parameters()) == 592_688
    assert torch.isfinite(output).all()
    assert torch.linalg.vector_norm(output, dim=-1).max().item() < 1.0


def test_matched_bc_is_deterministic():
    model = MatchedSequenceBC().eval()
    observation = torch.randn(2, 34)
    assert torch.equal(model(observation), model(observation))
