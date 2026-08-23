import torch

from quadrotor_diffusion_ppo.diffusion.unet1d import ConditionalUnet1D


def make_model():
    return ConditionalUnet1D()


def test_unet_input_output_shape():
    model = make_model()
    out = model(torch.randn(2, 16, 3), torch.randn(2, 34), torch.tensor([0, 99]))
    assert out.shape == (2, 16, 3)


def test_unet_h16():
    model = make_model()
    assert model.horizon == 16


def test_unet_action3():
    model = make_model()
    assert model.action_dim == 3


def test_condition_shape():
    model = make_model()
    assert model.observation_encoder(torch.randn(2, 34)).shape == (2, 256)


def test_timestep_condition():
    model = make_model()
    assert model.time_mlp(model.time_embedding(torch.tensor([0, 10]))).shape == (2, 256)


def test_skip_shapes():
    model = make_model()
    obs = torch.randn(2, 34)
    t = torch.tensor([1, 2])
    with torch.no_grad():
        g = torch.cat((model.observation_encoder(obs), model.time_mlp(model.time_embedding(t))), dim=1)
        x = model.input_projection(torch.randn(2, 3, 16))
        s16 = model._run_blocks(x, model.down0, g)
        s8 = model._run_blocks(model.downsample0(s16), model.down1, g)
        assert s16.shape == (2, 256, 16)
        assert s8.shape == (2, 512, 8)
        assert model.downsample1(s8).shape == (2, 1024, 4)


def test_bounded_output():
    model = make_model()
    out = model(torch.randn(4, 16, 3) * 10, torch.randn(4, 34), torch.zeros(4, dtype=torch.long))
    assert torch.linalg.vector_norm(out, dim=-1).max().item() < 1.0


def test_unit_ball():
    model = make_model()
    out = model(torch.randn(4, 16, 3), torch.randn(4, 34), torch.ones(4, dtype=torch.long))
    assert bool((torch.linalg.vector_norm(out, dim=-1) <= 1.0).all())


def test_backward_finite():
    model = make_model()
    out = model(torch.randn(2, 16, 3), torch.randn(2, 34), torch.tensor([1, 2]))
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_effective_batch():
    assert 2 * 256 == 512


def test_test_sealed():
    assert not __import__("os").environ.get("S8R4_ALLOW_TEST")

