from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from .fppool import FPPoolExplanation, FPPoolInnerExplanation


def draw_fppool_bit_highlight(
    mol: Chem.Mol, explanation: FPPoolInnerExplanation, image_size: tuple[int, int] = (300, 300)
) -> Image.Image:
    """Render one inner-level explanation with atom and bond highlights.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolInnerExplanation
        The inner-level explanation payload describing one highlighted bit.
    image_size : tuple[int, int], default=(300, 300)
        The rendered image size in pixels.

    Returns
    -------
    Image.Image
        A PIL image containing the highlighted molecule rendering.
    """
    if explanation.atom_weights:
        max_atom_weight = max(explanation.atom_weights)
    else:
        max_atom_weight = 1.0

    atom_colors = {}
    for atom_idx, atom_weight in zip(
        explanation.provenance.atom_indices, explanation.atom_weights, strict=True
    ):
        normalized = 0.0 if max_atom_weight == 0.0 else atom_weight / max_atom_weight
        atom_colors[atom_idx] = (1.0, 0.85 - 0.45 * normalized, 0.15)

    bond_color = (
        min(1.0, 0.25 + explanation.score),
        max(0.0, 0.55 - 0.2 * explanation.score),
        0.15,
    )
    bond_colors = {bond_idx: bond_color for bond_idx in explanation.provenance.bond_indices}
    return Draw.MolToImage(
        mol,
        size=image_size,
        highlightAtoms=list(explanation.provenance.atom_indices),
        highlightBonds=list(explanation.provenance.bond_indices),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )


def draw_fppool_inner_explanations(
    mol: Chem.Mol,
    explanation: FPPoolExplanation,
    top_k: int = 4,
    image_size: tuple[int, int] = (300, 300),
) -> list[Image.Image]:
    """Render the top-scoring inner-level explanations for one molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    top_k : int, default=4
        The maximum number of inner explanations to render.
    image_size : tuple[int, int], default=(300, 300)
        The rendered image size in pixels.

    Returns
    -------
    list[Image.Image]
        A ranked list of rendered PIL images.
    """
    ranked = sorted(
        explanation.inner_explanations,
        key=lambda inner_explanation: inner_explanation.score,
        reverse=True,
    )
    return [
        draw_fppool_bit_highlight(mol, inner_explanation, image_size=image_size)
        for inner_explanation in ranked[:top_k]
    ]


def plot_fppool_inter_summary(explanation: FPPoolExplanation, top_k_bits: int = 5) -> Figure:
    """Plot the strongest bit-level explanations grouped by fingerprint family.

    Parameters
    ----------
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    top_k_bits : int, default=5
        The maximum number of bit explanations to show per active family.

    Returns
    -------
    Figure
        A matplotlib figure summarizing inner bit scores grouped by family.
    """
    active_families = [
        inter_explanation
        for inter_explanation in explanation.inter_explanations
        if inter_explanation.is_active
    ]
    nrows = max(1, len(active_families))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3 * nrows))
    axes_list = [axes] if nrows == 1 else list(axes)

    if not active_families:
        axes_list[0].set_title("No active fingerprint families")
        axes_list[0].axis("off")
        return fig

    for axis, inter_explanation in zip(axes_list, active_families, strict=True):
        ranked = sorted(
            inter_explanation.bit_explanations,
            key=lambda bit_explanation: bit_explanation.score,
            reverse=True,
        )[:top_k_bits]
        labels = [
            f"{bit_explanation.provenance.family_name}:{bit_explanation.provenance.family_bit_index}"
            for bit_explanation in ranked
        ]
        values = [bit_explanation.score for bit_explanation in ranked]
        axis.barh(labels, values, color="#d97706")
        axis.set_title(
            f"{inter_explanation.family_name} (family weight={inter_explanation.family_weight:.3f})"
        )
        axis.set_xlabel("Inter x Global Score")
        axis.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_fppool_global_weights(explanation: FPPoolExplanation) -> Figure:
    """Plot global fingerprint-family weights for one molecule.

    Parameters
    ----------
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.

    Returns
    -------
    Figure
        A matplotlib figure summarizing family-level global weights.
    """
    fig, axis = plt.subplots(figsize=(6, 4))
    family_names = list(explanation.global_explanation.family_names)
    family_weights = list(explanation.global_explanation.family_weights)
    axis.bar(family_names, family_weights, color="#2563eb")
    axis.set_ylabel("Global Weight")
    axis.set_title(f"FPPool Global Weights (molecule {explanation.molecule_index})")
    fig.tight_layout()
    return fig
