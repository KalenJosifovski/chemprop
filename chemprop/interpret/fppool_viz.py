from __future__ import annotations

from io import BytesIO
from itertools import cycle

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from .fppool import FPPoolExplanation, FPPoolInnerExplanation, FPPoolInterExplanation


def _ensure_2d_coords(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with 2D coordinates available for drawing."""
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def _normalize_score_map(score_map: dict[int, float]) -> dict[int, float]:
    """Normalize non-negative scores to the ``[0, 1]`` range."""
    if not score_map:
        return {}

    max_score = max(score_map.values())
    if max_score <= 0.0:
        return {key: 0.0 for key in score_map}

    return {key: value / max_score for key, value in score_map.items()}


def _render_overlay_image(
    mol: Chem.Mol,
    atom_scores: dict[int, float],
    bond_scores: dict[int, float],
    image_size: tuple[int, int],
) -> Image.Image:
    """Render one molecule with scaled atom and bond overlays."""
    mol = _ensure_2d_coords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*image_size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.6
    options.highlightBondWidthMultiplier = 14
    options.atomHighlightsAreCircles = True

    normalized_atom_scores = _normalize_score_map(atom_scores)
    normalized_bond_scores = _normalize_score_map(bond_scores)

    highlight_atom_colors = {
        atom_idx: (0.22 + 0.12 * score, 0.45 + 0.25 * score, 0.30 + 0.10 * score)
        for atom_idx, score in normalized_atom_scores.items()
    }
    highlight_atom_radii = {
        atom_idx: 0.25 + 0.55 * score for atom_idx, score in normalized_atom_scores.items()
    }
    highlight_bond_colors = {
        bond_idx: (0.20 + 0.18 * score, 0.42 + 0.20 * score, 0.28 + 0.08 * score)
        for bond_idx, score in normalized_bond_scores.items()
    }

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=sorted(highlight_atom_colors),
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
        highlightBonds=sorted(highlight_bond_colors),
        highlightBondColors=highlight_bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _render_molecule_with_outlines(
    mol: Chem.Mol,
    image_size: tuple[int, int],
    outline_atom_indices: tuple[int, ...] = (),
    outline_bond_indices: tuple[int, ...] = (),
    outline_color: tuple[float, float, float] = (0.08, 0.24, 0.12),
) -> tuple[Image.Image, dict[int, tuple[float, float]]]:
    """Render a base molecule plus explicit substructure outlines and return atom draw coords."""
    mol = _ensure_2d_coords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*image_size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.fillHighlights = False
    options.continuousHighlight = False
    options.atomHighlightsAreCircles = False
    options.highlightBondWidthMultiplier = 18
    options.scaleHighlightBondWidth = True

    highlight_atoms = sorted(set(outline_atom_indices))
    highlight_bonds = sorted(set(outline_bond_indices))
    highlight_atom_colors = {atom_idx: outline_color for atom_idx in highlight_atoms}
    highlight_bond_colors = {bond_idx: outline_color for bond_idx in highlight_bonds}
    highlight_atom_radii = {atom_idx: 0.34 for atom_idx in highlight_atoms}

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
    )
    atom_draw_coords = {
        atom_idx: (drawer.GetDrawCoords(atom_idx).x, drawer.GetDrawCoords(atom_idx).y)
        for atom_idx in range(mol.GetNumAtoms())
    }
    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return image, atom_draw_coords


def _build_attention_field(
    atom_draw_coords: dict[int, tuple[float, float]],
    atom_scores: dict[int, float],
    image_size: tuple[int, int],
    sigma: float = 18.0,
) -> np.ndarray:
    """Build a smoothed atom-centered importance field over the rendered image canvas."""
    if not atom_scores:
        return np.zeros((image_size[1], image_size[0]), dtype=float)

    x_grid, y_grid = np.meshgrid(
        np.arange(image_size[0], dtype=float), np.arange(image_size[1], dtype=float)
    )
    field = np.zeros_like(x_grid, dtype=float)
    for atom_idx, score in atom_scores.items():
        x_coord, y_coord = atom_draw_coords[atom_idx]
        field += float(score) * np.exp(
            -((x_grid - x_coord) ** 2 + (y_grid - y_coord) ** 2) / (2.0 * sigma**2)
        )

    return field


def _render_attention_contours(
    base_image: Image.Image,
    atom_draw_coords: dict[int, tuple[float, float]],
    atom_scores: dict[int, float],
    image_size: tuple[int, int],
    contour_color: tuple[float, float, float] = (0.10, 0.42, 0.20),
    contour_cmap: LinearSegmentedColormap | None = None,
) -> Image.Image:
    """Overlay a smooth atom-importance contour field onto a rendered molecule image."""
    field = _build_attention_field(atom_draw_coords, atom_scores, image_size=image_size)
    fig, axis = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
    axis.imshow(base_image)
    axis.axis("off")

    if field.max() > 0.0:
        levels = np.linspace(field.max() * 0.2, field.max(), 6)
        cmap = contour_cmap or LinearSegmentedColormap.from_list(
            "fppool_single",
            [
                (1.0, 1.0, 1.0, 0.0),
                (*contour_color, 0.12),
                (*contour_color, 0.22),
                (*contour_color, 0.34),
            ],
        )
        axis.contourf(field, levels=levels, cmap=cmap, antialiased=True)
        axis.contour(field, levels=levels, colors=[contour_color], linewidths=0.9, alpha=0.35)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).resize(image_size)


def _render_attention_debug_contours(
    base_image: Image.Image,
    atom_draw_coords: dict[int, tuple[float, float]],
    atom_score_maps: tuple[dict[int, float], ...],
    image_size: tuple[int, int],
) -> Image.Image:
    """Overlay separate colored contour fields for multiple bit-level contributions."""
    fig, axis = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
    axis.imshow(base_image)
    axis.axis("off")

    colors = cycle(((0.07, 0.43, 0.20), (0.16, 0.37, 0.72), (0.72, 0.25, 0.18), (0.56, 0.28, 0.72)))
    for atom_scores, contour_color in zip(atom_score_maps, colors, strict=False):
        field = _build_attention_field(atom_draw_coords, atom_scores, image_size=image_size)
        if field.max() <= 0.0:
            continue
        levels = np.linspace(field.max() * 0.25, field.max(), 5)
        cmap = LinearSegmentedColormap.from_list(
            "fppool_debug",
            [
                (1.0, 1.0, 1.0, 0.0),
                (*contour_color, 0.10),
                (*contour_color, 0.18),
                (*contour_color, 0.28),
            ],
        )
        axis.contourf(field, levels=levels, cmap=cmap, antialiased=True)
        axis.contour(field, levels=levels, colors=[contour_color], linewidths=0.8, alpha=0.45)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).resize(image_size)


def _aggregate_inner_scores(
    inner_explanations: tuple[FPPoolInnerExplanation, ...]
) -> tuple[dict[int, float], dict[int, float]]:
    """Aggregate atom and bond scores over one or more inner explanations."""
    atom_scores: dict[int, float] = {}
    bond_scores: dict[int, float] = {}

    for inner_explanation in inner_explanations:
        bit_score = inner_explanation.score
        for atom_idx, atom_weight in zip(
            inner_explanation.provenance.atom_indices, inner_explanation.atom_weights, strict=True
        ):
            atom_scores[atom_idx] = atom_scores.get(atom_idx, 0.0) + float(atom_weight)

        for bond_idx in inner_explanation.provenance.bond_indices:
            bond_scores[bond_idx] = bond_scores.get(bond_idx, 0.0) + bit_score

    return atom_scores, bond_scores


def _weighted_inner_scores(
    inner_explanation: FPPoolInnerExplanation, level: str
) -> tuple[dict[int, float], dict[int, float]]:
    """Build atom and bond score maps for one inner explanation at a chosen hierarchy level."""
    if level == "inner":
        scale = 1.0
    elif level == "inter":
        scale = inner_explanation.inter_weight
    elif level == "global":
        scale = inner_explanation.inter_weight * inner_explanation.global_weight
    else:
        raise ValueError(f"Unsupported FPPool visualization level: {level}")

    atom_scores = {
        atom_idx: float(atom_weight) * scale
        for atom_idx, atom_weight in zip(
            inner_explanation.provenance.atom_indices, inner_explanation.atom_weights, strict=True
        )
    }
    bond_base = max(inner_explanation.atom_weights, default=0.0)
    bond_scores = {
        bond_idx: float(bond_base) * scale for bond_idx in inner_explanation.provenance.bond_indices
    }
    return atom_scores, bond_scores


def _merge_score_maps(
    score_maps: tuple[tuple[dict[int, float], dict[int, float]], ...]
) -> tuple[dict[int, float], dict[int, float]]:
    """Merge multiple atom/bond score maps by summation."""
    atom_scores: dict[int, float] = {}
    bond_scores: dict[int, float] = {}

    for atom_map, bond_map in score_maps:
        for atom_idx, score in atom_map.items():
            atom_scores[atom_idx] = atom_scores.get(atom_idx, 0.0) + score
        for bond_idx, score in bond_map.items():
            bond_scores[bond_idx] = bond_scores.get(bond_idx, 0.0) + score

    return atom_scores, bond_scores


def _active_inter_explanations(
    explanation: FPPoolExplanation,
) -> tuple[FPPoolInterExplanation, ...]:
    """Return the active fingerprint-family explanations in payload order."""
    return tuple(
        inter_explanation
        for inter_explanation in explanation.inter_explanations
        if inter_explanation.is_active and inter_explanation.bit_explanations
    )


def _filtered_inter_explanations(
    explanation: FPPoolExplanation, exclude_family_names: tuple[str, ...]
) -> tuple[FPPoolInterExplanation, ...]:
    """Return active inter explanations excluding requested fingerprint families."""
    return tuple(
        inter_explanation
        for inter_explanation in _active_inter_explanations(explanation)
        if inter_explanation.family_name not in exclude_family_names
    )


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
    atom_scores, _ = _weighted_inner_scores(explanation, level="inner")
    base_image, atom_draw_coords = _render_molecule_with_outlines(
        mol,
        image_size=image_size,
        outline_atom_indices=explanation.provenance.atom_indices,
        outline_bond_indices=explanation.provenance.bond_indices,
    )
    inner_cmap = LinearSegmentedColormap.from_list(
        "fppool_inner",
        [
            (1.0, 1.0, 1.0, 0.0),
            (0.22, 0.45, 0.95, 0.18),
            (0.55, 0.30, 0.85, 0.26),
            (0.90, 0.18, 0.22, 0.38),
        ],
    )
    return _render_attention_contours(
        base_image,
        atom_draw_coords,
        atom_scores,
        image_size,
        contour_color=(0.75, 0.10, 0.16),
        contour_cmap=inner_cmap,
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


def draw_fppool_inner_family_grid(
    mol: Chem.Mol,
    explanation: FPPoolExplanation,
    top_k_bits: int = 4,
    image_size: tuple[int, int] = (220, 220),
    exclude_family_names: tuple[str, ...] = (),
) -> Figure:
    """Render top-k inner explanations grouped by fingerprint family.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    top_k_bits : int, default=5
        The maximum number of bit explanations to show per active family.
    image_size : tuple[int, int], default=(220, 220)
        The rendered image size for each molecule panel.
    exclude_family_names : tuple[str, ...], default=()
        Optional fingerprint-family names to omit from the visualization.

    Returns
    -------
    Figure
        A matplotlib figure with one row per family and one column per ranked bit.
    """
    active_families = _filtered_inter_explanations(explanation, exclude_family_names)
    nrows = max(1, len(active_families))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=top_k_bits, figsize=(3 * top_k_bits, 3 * nrows), squeeze=False
    )

    if not active_families:
        axes[0, 0].set_title("No active fingerprint families")
        axes[0, 0].axis("off")
        return fig

    for row_index, inter_explanation in enumerate(active_families):
        ranked = sorted(
            inter_explanation.bit_explanations,
            key=lambda bit_explanation: bit_explanation.score,
            reverse=True,
        )[:top_k_bits]
        for col_index in range(top_k_bits):
            axis = axes[row_index, col_index]
            axis.axis("off")
            if col_index >= len(ranked):
                if col_index == 0:
                    axis.set_title(f"{inter_explanation.family_name}\n(no active bits)")
                continue

            bit_explanation = ranked[col_index]
            image = draw_fppool_bit_highlight(mol, bit_explanation, image_size=image_size)
            axis.imshow(image)
            axis.set_title(
                f"{inter_explanation.family_name}\nbit {bit_explanation.provenance.family_bit_index}\n"
                f"inner={bit_explanation.inner_weight_sum:.2f}"
            )

    fig.suptitle("Inner View", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fppool_inter_summary(
    mol: Chem.Mol,
    explanation: FPPoolExplanation,
    top_k_bits: int = 4,
    image_size: tuple[int, int] = (260, 260),
    exclude_family_names: tuple[str, ...] = (),
    color_bits_debug: bool = False,
) -> Figure:
    """Render one inter-level aggregated molecule overlay per active fingerprint family.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    top_k_bits : int, default=4
        The maximum number of inner explanations to aggregate per active family.
    image_size : tuple[int, int], default=(260, 260)
        The rendered image size for each family panel.
    exclude_family_names : tuple[str, ...], default=()
        Optional fingerprint-family names to omit from the visualization.
    color_bits_debug : bool, default=False
        Whether to overlay the top-k contributing bits with distinct contour colors instead of one
        combined family field.

    Returns
    -------
    Figure
        A matplotlib figure with one aggregated family overlay per active family.
    """
    active_families = _filtered_inter_explanations(explanation, exclude_family_names)
    nrows = max(1, len(active_families))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(4, 3.4 * nrows), squeeze=False)

    if not active_families:
        axes[0, 0].set_title("No active fingerprint families")
        axes[0, 0].axis("off")
        return fig

    for row_index, inter_explanation in enumerate(active_families):
        ranked = tuple(
            sorted(
                inter_explanation.bit_explanations,
                key=lambda bit_explanation: bit_explanation.score,
                reverse=True,
            )[:top_k_bits]
        )
        score_maps = tuple(
            _weighted_inner_scores(bit_explanation, level="inter") for bit_explanation in ranked
        )
        atom_scores, bond_scores = _merge_score_maps(score_maps)
        outline_atom_indices = tuple(
            atom_idx
            for bit_explanation in ranked
            for atom_idx in bit_explanation.provenance.atom_indices
        )
        outline_bond_indices = tuple(
            bond_idx
            for bit_explanation in ranked
            for bond_idx in bit_explanation.provenance.bond_indices
        )
        base_image, atom_draw_coords = _render_molecule_with_outlines(
            mol,
            image_size=image_size,
            outline_atom_indices=outline_atom_indices,
            outline_bond_indices=outline_bond_indices,
            outline_color=(0.18, 0.28, 0.20),
        )
        if color_bits_debug:
            image = _render_attention_debug_contours(
                base_image,
                atom_draw_coords,
                tuple(atom_map for atom_map, _ in score_maps),
                image_size,
            )
        else:
            image = _render_attention_contours(
                base_image, atom_draw_coords, atom_scores, image_size
            )

        axis = axes[row_index, 0]
        axis.imshow(image)
        axis.set_title(
            f"{inter_explanation.family_name} "
            f"(family weight={inter_explanation.family_weight:.3f}, top {min(top_k_bits, len(ranked))} bits)"
        )
        axis.axis("off")

    fig.suptitle("Inter View", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fppool_global_weights(
    mol: Chem.Mol,
    explanation: FPPoolExplanation,
    image_size: tuple[int, int] = (320, 320),
    exclude_family_names: tuple[str, ...] = (),
) -> Figure:
    """Render the global family-weighted overlay across all active fingerprint families.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    image_size : tuple[int, int], default=(320, 320)
        The rendered image size for the global panel.
    exclude_family_names : tuple[str, ...], default=()
        Optional fingerprint-family names to omit from the visualization.

    Returns
    -------
    Figure
        A matplotlib figure summarizing the globally aggregated molecule overlay.
    """
    ranked = tuple(
        sorted(
            (
                inner_explanation
                for inner_explanation in explanation.inner_explanations
                if inner_explanation.provenance.family_name not in exclude_family_names
            ),
            key=lambda inner_explanation: inner_explanation.score,
            reverse=True,
        )
    )
    score_maps = tuple(
        _weighted_inner_scores(inner_explanation, level="global") for inner_explanation in ranked
    )
    atom_scores, _ = _merge_score_maps(score_maps)
    base_image, atom_draw_coords = _render_molecule_with_outlines(mol, image_size=image_size)
    image = _render_attention_contours(base_image, atom_draw_coords, atom_scores, image_size)

    fig, axis = plt.subplots(figsize=(4.5, 4.5))
    axis.imshow(image)
    family_summary = ", ".join(
        f"{family_name}={family_weight:.2f}"
        for family_name, family_weight in zip(
            explanation.global_explanation.family_names,
            explanation.global_explanation.family_weights,
            strict=True,
        )
        if family_name not in exclude_family_names
    )
    axis.set_title(f"Global View\n{family_summary}")
    axis.axis("off")
    fig.tight_layout()
    return fig


def compose_fppool_multilevel_view(
    mol: Chem.Mol,
    explanation: FPPoolExplanation,
    top_k_inter: int = 4,
    image_size: tuple[int, int] = (300, 300),
    exclude_family_names: tuple[str, ...] = (),
    color_bits_debug: bool = False,
) -> Figure:
    """Compose a paper-style family-structured inner/inter/global view for one molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to render.
    explanation : FPPoolExplanation
        The full per-molecule explanation payload.
    top_k_inter : int, default=4
        The number of highest-scoring inner explanations to include in the inter panel.
    image_size : tuple[int, int], default=(300, 300)
        The rendered image size for each molecule panel.
    exclude_family_names : tuple[str, ...], default=()
        Optional fingerprint-family names to omit from the visualization.
    color_bits_debug : bool, default=False
        Whether to render the inter-family panels with distinct per-bit contour colors instead of
        one combined family field.

    Returns
    -------
    Figure
        A multi-panel figure with:

        - one row per family and ``top_k_inter`` columns for inner explanations
        - one aggregated inter-level overlay per family
        - one global overlay across all active families
    """
    active_families = _filtered_inter_explanations(explanation, exclude_family_names)
    if not active_families:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.set_title("No active FPPool explanations")
        axis.axis("off")
        return fig

    nrows = len(active_families)
    fig = plt.figure(figsize=(3 * top_k_inter + 8, 3.2 * nrows))
    outer = fig.add_gridspec(
        nrows=nrows, ncols=top_k_inter + 2, width_ratios=[1.0] * top_k_inter + [1.25, 1.4]
    )

    for row_index, inter_explanation in enumerate(active_families):
        ranked = tuple(
            sorted(
                inter_explanation.bit_explanations,
                key=lambda bit_explanation: bit_explanation.score,
                reverse=True,
            )[:top_k_inter]
        )
        for col_index in range(top_k_inter):
            axis = fig.add_subplot(outer[row_index, col_index])
            axis.axis("off")
            if col_index >= len(ranked):
                if col_index == 0:
                    axis.set_title(f"{inter_explanation.family_name}\n(no active bits)")
                continue

            bit_explanation = ranked[col_index]
            image = draw_fppool_bit_highlight(mol, bit_explanation, image_size=image_size)
            axis.imshow(np.asarray(image))
            axis.set_title(
                f"{inter_explanation.family_name}\nbit {bit_explanation.provenance.family_bit_index}",
                fontsize=9,
            )

        inter_axis = fig.add_subplot(outer[row_index, top_k_inter])
        inter_score_maps = tuple(
            _weighted_inner_scores(bit_explanation, level="inter") for bit_explanation in ranked
        )
        inter_atom_scores, _ = _merge_score_maps(inter_score_maps)
        inter_outline_atoms = tuple(
            atom_idx
            for bit_explanation in ranked
            for atom_idx in bit_explanation.provenance.atom_indices
        )
        inter_outline_bonds = tuple(
            bond_idx
            for bit_explanation in ranked
            for bond_idx in bit_explanation.provenance.bond_indices
        )
        inter_base_image, inter_atom_draw_coords = _render_molecule_with_outlines(
            mol,
            image_size=image_size,
            outline_atom_indices=inter_outline_atoms,
            outline_bond_indices=inter_outline_bonds,
            outline_color=(0.18, 0.28, 0.20),
        )
        if color_bits_debug:
            inter_image = _render_attention_debug_contours(
                inter_base_image,
                inter_atom_draw_coords,
                tuple(atom_map for atom_map, _ in inter_score_maps),
                image_size,
            )
        else:
            inter_image = _render_attention_contours(
                inter_base_image, inter_atom_draw_coords, inter_atom_scores, image_size
            )
        inter_axis.imshow(np.asarray(inter_image))
        inter_axis.set_title(f"{inter_explanation.family_name}\ninter", fontsize=9)
        inter_axis.axis("off")

    global_axis = fig.add_subplot(outer[:, top_k_inter + 1])
    global_score_maps = tuple(
        _weighted_inner_scores(inner_explanation, level="global")
        for inner_explanation in explanation.inner_explanations
        if inner_explanation.provenance.family_name not in exclude_family_names
    )
    global_atom_scores, _ = _merge_score_maps(global_score_maps)
    global_size = (image_size[0], image_size[1] * nrows)
    global_base_image, global_atom_draw_coords = _render_molecule_with_outlines(
        mol, image_size=global_size
    )
    global_image = _render_attention_contours(
        global_base_image, global_atom_draw_coords, global_atom_scores, image_size=global_size
    )
    global_axis.imshow(np.asarray(global_image))
    global_axis.set_title("global", fontsize=10)
    global_axis.axis("off")

    fig.text(0.18, 0.98, "Inner View", ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.77, 0.98, "Inter View", ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.91, 0.98, "Global View", ha="center", va="top", fontsize=14, fontweight="bold")

    inner_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    inner_sm = cm.ScalarMappable(norm=inner_norm, cmap=cm.get_cmap("coolwarm"))
    inner_sm.set_array([])
    colorbar_axis = fig.add_axes((0.11, 0.02, 0.20, 0.02))
    colorbar = fig.colorbar(inner_sm, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Inner atom importance", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    legend_line = Line2D([0], [0], color=(0.10, 0.42, 0.20), lw=2)
    fig.legend(
        [legend_line],
        ["Inter/global contour intensity"],
        loc="lower center",
        bbox_to_anchor=(0.63, 0.01),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig
