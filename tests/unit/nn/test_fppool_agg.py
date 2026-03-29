from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from chemprop.data import FPPoolBatch
from chemprop.models import MPNN
from chemprop.nn import FPPoolAggregation, RegressionFFN


def make_fppool_batch(atom_fp: Tensor) -> FPPoolBatch:
    return FPPoolBatch(
        atom_fp=atom_fp.bool(),
        fp_family_lengths=torch.tensor([2, 2], dtype=torch.long),
        fp_family_names=["morgan", "rdkit"],
        molecule_atom_slices=torch.tensor([0, 3, 5], dtype=torch.long),
    )


def make_atoms_repr_fppool_batch(atom_fp: Tensor) -> FPPoolBatch:
    return FPPoolBatch(
        atom_fp=atom_fp.bool(),
        fp_family_lengths=torch.tensor([1, 3], dtype=torch.long),
        fp_family_names=["atoms", "morgan"],
        molecule_atom_slices=torch.tensor([0, 3, 5], dtype=torch.long),
    )


def test_fppool_aggregation_requires_context():
    agg = FPPoolAggregation()
    H = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="requires an FPPoolBatch"):
        agg(H, batch)


def test_fppool_aggregation_returns_expected_shape_and_attention_state():
    agg = FPPoolAggregation()
    H = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 1.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    atom_fp = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.bool
    )
    fppool_batch = make_fppool_batch(atom_fp)

    pooled = agg(H, batch, fppool_batch=fppool_batch)

    assert pooled.shape == (2, 4)
    assert torch.isfinite(pooled).all()
    assert agg.last_attention_state is not None
    assert agg.last_attention_state.global_weights.shape == (2, 2)
    assert agg.last_attention_state.global_mask.shape == (2, 2)
    assert len(agg.last_attention_state.inter_weights) == 2
    assert agg.last_attention_state.inner_weights.numel() == int(atom_fp.sum().item())


def test_fppool_family_slicing_and_zero_membership_groups_are_safe():
    agg = FPPoolAggregation()
    H = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    atom_fp = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool
    )
    fppool_batch = make_fppool_batch(atom_fp)

    pooled = agg(H, batch, fppool_batch=fppool_batch)

    assert torch.isfinite(pooled).all()
    assert pooled.shape == (2, 4)
    assert torch.equal(
        agg.last_attention_state.global_mask,
        torch.tensor([[True, False], [True, False]], dtype=torch.bool),
    )
    assert not agg.last_attention_state.inter_masks[1].any()


def test_fppool_aggregation_consumes_atoms_repr_as_leading_family():
    agg = FPPoolAggregation()
    H = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    atom_fp = torch.tensor(
        [[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool
    )
    fppool_batch = make_atoms_repr_fppool_batch(atom_fp)

    pooled = agg(H, batch, fppool_batch=fppool_batch)

    assert pooled.shape == (2, 4)
    assert torch.isfinite(pooled).all()
    assert agg.last_attention_state is not None
    assert agg.last_attention_state.global_mask.shape == (2, 2)
    assert agg.last_attention_state.inter_masks[0].shape == (2, 1)
    assert agg.last_attention_state.inter_masks[0].all()


@dataclass
class DummyBatchMolGraph:
    V: Tensor
    batch: Tensor


class DummyMessagePassing(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.hparams = {"cls": self.__class__, "output_dim": output_dim}

    def forward(self, bmg: DummyBatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        return bmg.V


def test_mpnn_fingerprint_passes_fppool_batch_to_aggregation():
    H = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 1.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    atom_fp = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.bool
    )
    fppool_batch = make_fppool_batch(atom_fp)
    bmg = DummyBatchMolGraph(V=H, batch=batch)

    agg = FPPoolAggregation()
    predictor = RegressionFFN(input_dim=4)
    model = MPNN(DummyMessagePassing(output_dim=4), agg, predictor)

    expected = agg(H, batch, fppool_batch=fppool_batch)
    observed = model.fingerprint(bmg, fppool_batch=fppool_batch)

    torch.testing.assert_close(observed, expected)
