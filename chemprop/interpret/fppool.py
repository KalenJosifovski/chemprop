from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem

from chemprop.data import FPPoolBatch, FPPoolConfig, build_fppool_molecule_provenance
from chemprop.nn import FPPoolAggregation


@dataclass(frozen=True, slots=True)
class FPPoolBitProvenance:
    """Public provenance metadata for one active FPPool bit."""

    family_name: str
    family_index: int
    family_bit_index: int
    global_bit_index: int
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FPPoolInnerExplanation:
    """Inner-level explanation for one active bit of one molecule."""

    provenance: FPPoolBitProvenance
    atom_weights: tuple[float, ...]
    inner_weight_sum: float
    inter_weight: float
    global_weight: float
    score: float


@dataclass(frozen=True, slots=True)
class FPPoolInterExplanation:
    """Inter-level explanation for one fingerprint family of one molecule."""

    family_name: str
    family_index: int
    family_weight: float
    is_active: bool
    bit_explanations: tuple[FPPoolInnerExplanation, ...]


@dataclass(frozen=True, slots=True)
class FPPoolGlobalExplanation:
    """Global family-level explanation for one molecule."""

    family_names: tuple[str, ...]
    family_weights: tuple[float, ...]
    family_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class FPPoolExplanation:
    """Complete FPPool explanation payload for one molecule."""

    molecule_index: int
    inner_explanations: tuple[FPPoolInnerExplanation, ...]
    inter_explanations: tuple[FPPoolInterExplanation, ...]
    global_explanation: FPPoolGlobalExplanation


def _resolve_fppool_aggregation(model_or_aggregation: object) -> FPPoolAggregation:
    if isinstance(model_or_aggregation, FPPoolAggregation):
        return model_or_aggregation

    agg = getattr(model_or_aggregation, "agg", None)
    if isinstance(agg, FPPoolAggregation):
        return agg

    raise ValueError("Expected an FPPoolAggregation or a model exposing one at `.agg`.")


def _build_inner_weight_map(
    fppool_batch: FPPoolBatch, aggregation: FPPoolAggregation
) -> dict[tuple[int, int], dict[int, float]]:
    state = aggregation.last_attention_state
    if state is None:
        raise ValueError("No FPPool attention state is available. Run a forward pass first.")

    molecule_atom_slices = fppool_batch.molecule_atom_slices.detach().cpu()
    upper_bounds = molecule_atom_slices[1:]
    atom_indices = state.inner_atom_indices.detach().cpu()
    bit_indices = state.inner_bit_indices.detach().cpu()
    weights = state.inner_weights.detach().cpu()
    molecule_indices = torch.bucketize(atom_indices, upper_bounds, right=False)

    inner_weight_map: dict[tuple[int, int], dict[int, float]] = {}
    for atom_idx, bit_idx, weight, molecule_index in zip(
        atom_indices.tolist(),
        bit_indices.tolist(),
        weights.tolist(),
        molecule_indices.tolist(),
        strict=True,
    ):
        atom_start = int(molecule_atom_slices[molecule_index].item())
        local_atom_idx = atom_idx - atom_start
        key = (molecule_index, bit_idx)
        inner_weight_map.setdefault(key, {})[local_atom_idx] = float(weight)

    return inner_weight_map


def extract_fppool_explanation(
    model_or_aggregation: object,
    mols: Sequence[Chem.Mol],
    fppool_batch: FPPoolBatch,
    config: FPPoolConfig,
    top_k_bits: int | None = None,
) -> tuple[FPPoolExplanation, ...]:
    """Extract public inner/inter/global FPPool explanations from the latest forward pass.

    Parameters
    ----------
    model_or_aggregation : object
        Either an :class:`~chemprop.nn.FPPoolAggregation` instance or a model exposing one at
        ``.agg``.
    mols : Sequence[Chem.Mol]
        The molecules aligned to the batched FPPool forward pass.
    fppool_batch : FPPoolBatch
        The batch metadata used in the forward pass.
    config : FPPoolConfig
        The configuration needed to rebuild explicit atom-and-bond provenance.
    top_k_bits : int | None, default=None
        Optional truncation applied independently within each family after sorting bit explanations
        by decreasing score.

    Returns
    -------
    tuple[FPPoolExplanation, ...]
        One explanation payload per molecule in the batch.

    Raises
    ------
    ValueError
        If the aggregation is not FPPool, if no attention state is available, or if the provided
        molecules/config are inconsistent with the batch metadata.
    """
    aggregation = _resolve_fppool_aggregation(model_or_aggregation)
    state = aggregation.last_attention_state
    if state is None:
        raise ValueError("No FPPool attention state is available. Run a forward pass first.")

    family_names = tuple(fppool_batch.fp_family_names)
    family_lengths = tuple(
        int(length) for length in fppool_batch.fp_family_lengths.detach().cpu().tolist()
    )
    if family_names != tuple(config.active_family_names):
        raise ValueError(
            "FPPool batch family names do not match the requested interpretation config."
        )
    if family_lengths != tuple(int(length) for length in config.family_lengths.tolist()):
        raise ValueError(
            "FPPool batch family lengths do not match the requested interpretation config."
        )
    if len(mols) != int(fppool_batch.molecule_atom_slices.numel() - 1):
        raise ValueError("Number of molecules must match the batched FPPool molecule count.")

    inner_weight_map = _build_inner_weight_map(fppool_batch, aggregation)
    global_weights = state.global_weights.detach().cpu()
    global_mask = state.global_mask.detach().cpu()

    explanations: list[FPPoolExplanation] = []
    for molecule_index, mol in enumerate(mols):
        provenance = build_fppool_molecule_provenance(mol, config)
        membership_map = provenance.membership_map()
        family_weight_row = global_weights[molecule_index]
        family_mask_row = global_mask[molecule_index]

        inter_explanations: list[FPPoolInterExplanation] = []
        inner_explanations: list[FPPoolInnerExplanation] = []
        bit_offset = 0
        for family_index, (family_name, family_length) in enumerate(
            zip(family_names, family_lengths, strict=True)
        ):
            family_weight = float(family_weight_row[family_index].item())
            family_active = bool(family_mask_row[family_index].item())
            family_inter_weights = state.inter_weights[family_index][molecule_index].detach().cpu()
            family_inter_mask = state.inter_masks[family_index][molecule_index].detach().cpu()

            family_bit_explanations: list[FPPoolInnerExplanation] = []
            for family_bit_index in range(family_length):
                if not bool(family_inter_mask[family_bit_index].item()):
                    continue

                global_bit_index = bit_offset + family_bit_index
                membership = membership_map.get(global_bit_index)
                if membership is None:
                    continue

                bit_provenance = FPPoolBitProvenance(
                    family_name=membership.family_name,
                    family_index=membership.family_index,
                    family_bit_index=membership.family_bit_index,
                    global_bit_index=membership.global_bit_index,
                    atom_indices=membership.atom_indices,
                    bond_indices=membership.bond_indices,
                )
                atom_weight_lookup = inner_weight_map.get((molecule_index, global_bit_index), {})
                atom_weights = tuple(
                    float(atom_weight_lookup.get(atom_idx, 0.0))
                    for atom_idx in membership.atom_indices
                )
                inter_weight = float(family_inter_weights[family_bit_index].item())
                explanation = FPPoolInnerExplanation(
                    provenance=bit_provenance,
                    atom_weights=atom_weights,
                    inner_weight_sum=float(sum(atom_weights)),
                    inter_weight=inter_weight,
                    global_weight=family_weight,
                    score=inter_weight * family_weight,
                )
                family_bit_explanations.append(explanation)

            family_bit_explanations.sort(key=lambda explanation: explanation.score, reverse=True)
            if top_k_bits is not None:
                family_bit_explanations = family_bit_explanations[:top_k_bits]

            inter_explanation = FPPoolInterExplanation(
                family_name=family_name,
                family_index=family_index,
                family_weight=family_weight,
                is_active=family_active,
                bit_explanations=tuple(family_bit_explanations),
            )
            inter_explanations.append(inter_explanation)
            inner_explanations.extend(family_bit_explanations)
            bit_offset += family_length

        inner_explanations.sort(key=lambda explanation: explanation.score, reverse=True)
        explanations.append(
            FPPoolExplanation(
                molecule_index=molecule_index,
                inner_explanations=tuple(inner_explanations),
                inter_explanations=tuple(inter_explanations),
                global_explanation=FPPoolGlobalExplanation(
                    family_names=family_names,
                    family_weights=tuple(float(weight) for weight in family_weight_row.tolist()),
                    family_mask=tuple(bool(mask) for mask in family_mask_row.tolist()),
                ),
            )
        )

    return tuple(explanations)
