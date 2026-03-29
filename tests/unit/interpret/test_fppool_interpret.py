from matplotlib.figure import Figure
from PIL import Image
import torch

from chemprop.data import FPPoolBatch, FPPoolConfig, build_fppool_atom_fp
from chemprop.interpret import (
    draw_fppool_bit_highlight,
    draw_fppool_inner_family_grid,
    draw_fppool_inner_explanations,
    extract_fppool_explanation,
    plot_fppool_global_weights,
    plot_fppool_inter_summary,
)
from chemprop.nn import FPPoolAggregation
from chemprop.utils import make_mol


def _make_real_fppool_batch(mols, config: FPPoolConfig) -> FPPoolBatch:
    atom_fps = [build_fppool_atom_fp(mol, config) for mol in mols]
    molecule_atom_slices = [0]
    for atom_fp in atom_fps:
        molecule_atom_slices.append(molecule_atom_slices[-1] + atom_fp.shape[0])

    return FPPoolBatch(
        atom_fp=torch.from_numpy(
            torch.concatenate([torch.from_numpy(atom_fp) for atom_fp in atom_fps], dim=0).numpy()
        ).bool(),
        fp_family_lengths=torch.tensor(config.family_lengths.tolist(), dtype=torch.long),
        fp_family_names=config.active_family_names,
        molecule_atom_slices=torch.tensor(molecule_atom_slices, dtype=torch.long),
    )


def test_extract_fppool_explanation_returns_inner_inter_and_global_payloads():
    mols = [
        make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False),
        make_mol("CC", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False),
    ]
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=True)
    fppool_batch = _make_real_fppool_batch(mols, config)
    H = torch.randn(fppool_batch.atom_fp.shape[0], 8)
    batch = torch.tensor(
        [0] * mols[0].GetNumAtoms() + [1] * mols[1].GetNumAtoms(), dtype=torch.long
    )

    aggregation = FPPoolAggregation()
    pooled = aggregation(H, batch, fppool_batch=fppool_batch)
    explanations = extract_fppool_explanation(aggregation, mols, fppool_batch, config, top_k_bits=3)

    assert pooled.shape == (2, 8)
    assert len(explanations) == 2
    assert explanations[0].global_explanation.family_names == ("atoms", "morgan")
    assert explanations[0].global_explanation.family_mask == (True, True)
    assert len(explanations[0].inter_explanations) == 2
    assert explanations[0].inner_explanations
    top_inner = explanations[0].inner_explanations[0]
    assert top_inner.score >= 0.0
    assert len(top_inner.provenance.atom_indices) == len(top_inner.atom_weights)
    assert top_inner.provenance.family_name in {"atoms", "morgan"}


def test_draw_and_plot_helpers_return_renderable_objects():
    mols = [make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)]
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=True)
    fppool_batch = _make_real_fppool_batch(mols, config)
    H = torch.randn(fppool_batch.atom_fp.shape[0], 8)
    batch = torch.zeros(fppool_batch.atom_fp.shape[0], dtype=torch.long)

    aggregation = FPPoolAggregation()
    aggregation(H, batch, fppool_batch=fppool_batch)
    explanation = extract_fppool_explanation(aggregation, mols, fppool_batch, config, top_k_bits=2)[
        0
    ]

    image = draw_fppool_bit_highlight(
        mols[0], explanation.inner_explanations[0], image_size=(200, 200)
    )
    gallery = draw_fppool_inner_explanations(mols[0], explanation, top_k=2, image_size=(150, 150))
    inner_grid_fig = draw_fppool_inner_family_grid(mols[0], explanation, top_k_bits=2)
    inter_fig = plot_fppool_inter_summary(
        mols[0], explanation, top_k_bits=2, exclude_family_names=("atoms",)
    )
    global_fig = plot_fppool_global_weights(mols[0], explanation, exclude_family_names=("atoms",))

    assert isinstance(image, Image.Image)
    assert image.size == (200, 200)
    assert len(gallery) == min(2, len(explanation.inner_explanations))
    assert all(isinstance(rendered, Image.Image) for rendered in gallery)
    assert isinstance(inner_grid_fig, Figure)
    assert isinstance(inter_fig, Figure)
    assert isinstance(global_fig, Figure)


def test_multilevel_helpers_can_exclude_atoms_family():
    mols = [make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)]
    config = FPPoolConfig(
        family_names=("morgan", "rdkit", "pubchem"),
        morgan_nbits=32,
        rdkit_nbits=64,
        atoms_repr=True,
    )
    fppool_batch = _make_real_fppool_batch(mols, config)
    H = torch.randn(fppool_batch.atom_fp.shape[0], 8)
    batch = torch.zeros(fppool_batch.atom_fp.shape[0], dtype=torch.long)

    aggregation = FPPoolAggregation()
    aggregation(H, batch, fppool_batch=fppool_batch)
    explanation = extract_fppool_explanation(aggregation, mols, fppool_batch, config, top_k_bits=2)[
        0
    ]

    inner_grid_fig = draw_fppool_inner_family_grid(
        mols[0], explanation, top_k_bits=2, exclude_family_names=("atoms",)
    )
    inter_fig = plot_fppool_inter_summary(
        mols[0], explanation, top_k_bits=2, exclude_family_names=("atoms",)
    )
    global_fig = plot_fppool_global_weights(mols[0], explanation, exclude_family_names=("atoms",))

    assert isinstance(inner_grid_fig, Figure)
    assert isinstance(inter_fig, Figure)
    assert isinstance(global_fig, Figure)


def test_extract_fppool_explanation_supports_multifamily_batches():
    mols = [
        make_mol("c1ccccc1O", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)
    ]
    config = FPPoolConfig(
        family_names=("morgan", "rdkit", "pubchem"),
        morgan_nbits=32,
        rdkit_nbits=64,
        atoms_repr=True,
    )
    fppool_batch = _make_real_fppool_batch(mols, config)
    H = torch.randn(fppool_batch.atom_fp.shape[0], 8)
    batch = torch.zeros(fppool_batch.atom_fp.shape[0], dtype=torch.long)

    aggregation = FPPoolAggregation()
    aggregation(H, batch, fppool_batch=fppool_batch)
    explanation = extract_fppool_explanation(aggregation, mols, fppool_batch, config, top_k_bits=2)[
        0
    ]

    assert explanation.global_explanation.family_names == ("atoms", "morgan", "rdkit", "pubchem")
    assert len(explanation.inter_explanations) == 4
    assert any(inter.family_name == "rdkit" for inter in explanation.inter_explanations)
    assert any(inter.family_name == "pubchem" for inter in explanation.inter_explanations)
