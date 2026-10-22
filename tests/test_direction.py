"""Paired direction tests: H&E -> DAPI and DAPI -> H&E share one trainer."""
import pytest
import torch

from src.models.train_paired import UNetGenerator, prepare_pair


def test_he2dapi_pair():
    he = torch.zeros(2, 3, 4, 4)
    dapi = torch.ones(2, 1, 4, 4) * 0.5
    src, src01, target = prepare_pair({"he": he, "dapi": dapi}, "he2dapi")
    assert src.shape == he.shape
    assert src01.shape == he.shape
    assert target.shape == dapi.shape
    assert torch.allclose(src01, torch.full_like(he, 0.5))


def test_dapi2he_pair():
    he = torch.zeros(2, 3, 4, 4)
    dapi = torch.ones(2, 1, 4, 4) * 0.5
    src, src01, target = prepare_pair({"he": he, "dapi": dapi}, "dapi2he")
    assert src.shape == dapi.shape
    assert src01.shape == dapi.shape
    assert target.shape == he.shape
    assert torch.allclose(src, torch.zeros_like(dapi))
    assert torch.allclose(target, torch.full_like(he, 0.5))


def test_unknown_direction_rejected():
    batch = {"he": torch.zeros(1, 3, 4, 4), "dapi": torch.zeros(1, 1, 4, 4)}
    with pytest.raises(ValueError):
        prepare_pair(batch, "sideways")


def test_reverse_generator_channels():
    G = UNetGenerator(in_ch=1, out_ch=3, ngf=4)
    out = G(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 3, 32, 32)
