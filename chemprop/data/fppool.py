from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from importlib.resources import as_file, files
import json
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint

from chemprop.data.datapoints import MoleculeDatapoint

PUBCHEM_NBITS = 881
PUBCHEM_SMARTS_BIT_IDS = tuple(range(115)) + tuple(range(263, PUBCHEM_NBITS))
SUPPORTED_FPPOOL_FAMILIES = frozenset({"morgan", "rdkit", "pubchem"})


@dataclass(frozen=True, slots=True)
class FPPoolBitMembership:
    """Atom-and-bond membership metadata for one concatenated FPPool bit.

    Parameters
    ----------
    family_name : str
        The fingerprint family name associated with this membership.
    family_index : int
        The index of the fingerprint family in the concatenated family ordering.
    family_bit_index : int
        The bit index within the local fingerprint family.
    global_bit_index : int
        The bit index within the full concatenated FPPool matrix.
    atom_indices : tuple[int, ...]
        The ordered atom indices participating in the bit membership.
    bond_indices : tuple[int, ...]
        The ordered bond indices participating in the bit membership.
    """

    family_name: str
    family_index: int
    family_bit_index: int
    global_bit_index: int
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FPPoolMoleculeProvenance:
    """Per-molecule FPPool provenance for interpretability and visualization.

    Parameters
    ----------
    family_names : tuple[str, ...]
        The ordered fingerprint family names used for the concatenated FPPool matrix.
    family_lengths : tuple[int, ...]
        The ordered bit counts for the active fingerprint families.
    bit_memberships : tuple[FPPoolBitMembership, ...]
        The active bit memberships keyed by their concatenated bit indices.
    """

    family_names: tuple[str, ...]
    family_lengths: tuple[int, ...]
    bit_memberships: tuple[FPPoolBitMembership, ...]

    def membership_map(self) -> dict[int, FPPoolBitMembership]:
        """Return the active memberships keyed by global bit index."""
        return {membership.global_bit_index: membership for membership in self.bit_memberships}


@dataclass(frozen=True, slots=True)
class FPPoolConfig:
    """Configuration for FPPool atom-membership generation.

    Parameters
    ----------
    family_names : tuple[str, ...]
        The ordered fingerprint family names used to construct the concatenated membership matrix.
    morgan_nbits : int, default=1024
        The number of folded Morgan fingerprint bits.
    morgan_radius : int, default=2
        The Morgan fingerprint radius.
    rdkit_nbits : int, default=1024
        The number of folded RDKit path fingerprint bits.
    rdkit_min_path : int, default=1
        The minimum path length used for RDKit path fingerprints.
    rdkit_max_path : int, default=5
        The maximum path length used for RDKit path fingerprints.
    atoms_repr : bool, default=True
        Whether to prepend the synthetic atomic special family used by the reference FPPool
        implementation. When enabled, this contributes a leading all-atoms membership column.
    schema_version : str, default="2"
        A small cache-schema discriminator used as part of cache identity and metadata.
    """

    family_names: tuple[str, ...] = ("morgan",)
    morgan_nbits: int = 1024
    morgan_radius: int = 2
    rdkit_nbits: int = 1024
    rdkit_min_path: int = 1
    rdkit_max_path: int = 5
    atoms_repr: bool = True
    schema_version: str = "2"

    def __post_init__(self) -> None:
        if len(self.family_names) == 0:
            raise ValueError("FPPool must include at least one non-synthetic fingerprint family.")
        if len(set(self.family_names)) != len(self.family_names):
            raise ValueError("FPPool family names must be unique.")

        unsupported = [name for name in self.family_names if name not in SUPPORTED_FPPOOL_FAMILIES]
        if unsupported:
            raise ValueError(
                "Unsupported FPPool family/families: "
                + ", ".join(repr(name) for name in unsupported)
            )
        if self.morgan_nbits <= 0:
            raise ValueError("Morgan fingerprint bit count must be positive.")
        if self.morgan_radius < 0:
            raise ValueError("Morgan fingerprint radius must be non-negative.")
        if self.rdkit_nbits <= 0:
            raise ValueError("RDKit fingerprint bit count must be positive.")
        if self.rdkit_min_path <= 0:
            raise ValueError("RDKit minimum path length must be positive.")
        if self.rdkit_max_path < self.rdkit_min_path:
            raise ValueError("RDKit maximum path length must be at least the minimum path length.")

    @property
    def family_lengths(self) -> np.ndarray:
        """The ordered family bit counts for the active FPPool families."""
        lengths = []
        if self.atoms_repr:
            lengths.append(1)
        for family_name in self.family_names:
            if family_name == "morgan":
                lengths.append(self.morgan_nbits)
            elif family_name == "rdkit":
                lengths.append(self.rdkit_nbits)
            elif family_name == "pubchem":
                lengths.append(PUBCHEM_NBITS)
            else:
                raise ValueError(f"Unsupported FPPool family '{family_name}'.")

        return np.asarray(lengths, dtype=int)

    @property
    def active_family_names(self) -> list[str]:
        """The ordered active FPPool family names."""
        family_names = list(self.family_names)
        if self.atoms_repr:
            return ["atoms", *family_names]
        return family_names

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["family_names"] = self.active_family_names
        metadata["family_lengths"] = self.family_lengths.tolist()
        return metadata


def derive_fppool_cache_key(dataset_path: PathLike, config: FPPoolConfig) -> str:
    """Build a stable cache key from the dataset path and FPPool configuration."""
    payload = {"dataset_path": str(Path(dataset_path).resolve()), "config": config.to_metadata()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def derive_fppool_cache_dir(
    dataset_path: PathLike, config: FPPoolConfig, cache_root: PathLike | None = None
) -> Path:
    """Return the cache directory for a dataset/config pair."""
    root = Path(".cache") / "fppool" if cache_root is None else Path(cache_root)
    return root / derive_fppool_cache_key(dataset_path, config)


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def _atom_fp_path(cache_dir: Path) -> Path:
    return cache_dir / "atom_fp.npz"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _save_npz_atomic(path: Path, atom_fps: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez(tmp_path, *atom_fps)
    tmp_path.replace(path)


def save_fppool_cache(
    cache_dir: PathLike, atom_fps: list[np.ndarray], dataset_path: PathLike, config: FPPoolConfig
) -> None:
    """Save a directory-based FPPool cache for local single-process use."""
    cache_dir = Path(cache_dir)
    metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "family_names": config.active_family_names,
        "family_lengths": config.family_lengths.tolist(),
        "config": config.to_metadata(),
        "num_datapoints": len(atom_fps),
    }
    _write_json_atomic(_meta_path(cache_dir), metadata)
    _save_npz_atomic(_atom_fp_path(cache_dir), atom_fps)


def load_fppool_cache(
    cache_dir: PathLike, dataset_path: PathLike, config: FPPoolConfig
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load a cached set of atom-membership arrays and validate core metadata."""
    cache_dir = Path(cache_dir)
    with _meta_path(cache_dir).open(encoding="utf-8") as fh:
        metadata = json.load(fh)

    expected_dataset_path = str(Path(dataset_path).resolve())
    if metadata["dataset_path"] != expected_dataset_path:
        raise ValueError("Cached FPPool dataset path does not match the requested dataset path.")

    if metadata["config"] != config.to_metadata():
        raise ValueError("Cached FPPool configuration does not match the requested configuration.")

    family_names = list(metadata["family_names"])
    family_lengths = np.asarray(metadata["family_lengths"], dtype=int)

    if family_names != config.active_family_names:
        raise ValueError("Cached FPPool family names do not match the requested configuration.")
    if not np.array_equal(family_lengths, config.family_lengths):
        raise ValueError("Cached FPPool family lengths do not match the requested configuration.")

    loaded = np.load(_atom_fp_path(cache_dir))
    atom_fps = [
        np.asarray(loaded[f"arr_{idx}"], dtype=bool) for idx in range(metadata["num_datapoints"])
    ]

    return atom_fps, family_lengths, family_names


def morgan_bit_environment_atom_indices(
    mol: Chem.Mol, atom_idx: int, radius: int
) -> tuple[int, ...]:
    """Return the atom indices participating in a Morgan bit environment."""
    bond_environment = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atom_indices = {atom_idx}
    for bond_idx in bond_environment:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())

    return tuple(sorted(atom_indices))


def morgan_bit_environment_bond_indices(
    mol: Chem.Mol, atom_idx: int, radius: int
) -> tuple[int, ...]:
    """Return the bond indices participating in a Morgan bit environment."""
    bond_environment = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    return tuple(sorted(bond_environment))


def rdkit_bit_path_atom_bond_indices(
    mol: Chem.Mol, bond_indices: Iterable[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the atom and bond indices participating in an RDKit path fingerprint bit."""
    unique_bond_indices = tuple(sorted(set(int(bond_idx) for bond_idx in bond_indices)))
    atom_indices: set[int] = set()
    for bond_idx in unique_bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())

    return tuple(sorted(atom_indices)), unique_bond_indices


def _pubchem_data_path():
    return files("chemprop").joinpath("data/assets/pubchembit.pkl")


@lru_cache(maxsize=1)
def _load_pubchem_bits() -> pd.DataFrame:
    """Load the packaged PubChem SMARTS/rule metadata asset."""
    with as_file(_pubchem_data_path()) as pubchem_path:
        return pd.read_pickle(pubchem_path)


@lru_cache(maxsize=1)
def _pubchem_keys() -> tuple[tuple[Chem.Mol, int], ...]:
    df_bits = _load_pubchem_bits()
    return tuple(df_bits.patt.dropna().tolist())


def _pubchem_atoms_to_bonds(mol: Chem.Mol, atom_indices: Iterable[int]) -> tuple[int, ...]:
    atom_index_set = set(int(atom_idx) for atom_idx in atom_indices)
    bond_indices = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_index_set and bond.GetEndAtomIdx() in atom_index_set
    ]
    return tuple(sorted(set(bond_indices)))


def _pubchem_map_ring_counts(
    counts: dict[int, int],
    memberships: dict[int, tuple[tuple[int, ...], ...]],
    bits: np.ndarray,
    bit_info: dict[int, tuple[tuple[int, ...], ...]],
    offset: int = 0,
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    ring_specs = {
        3: ([0, 7], [1, 2]),
        4: ([14, 21], [1, 2]),
        5: ([28, 35, 42, 49, 56], [1, 2, 3, 4, 5]),
        6: ([63, 70, 77, 84, 91], [1, 2, 3, 4, 5]),
        7: ([98, 105], [1, 2]),
        8: ([112, 119], [1, 2]),
        9: ([126], [1]),
        10: ([133], [1]),
    }

    for ring_size, (bit_ids, thresholds) in ring_specs.items():
        for threshold, bit_idx in zip(thresholds, bit_ids, strict=True):
            if counts[ring_size] >= threshold:
                global_bit_idx = bit_idx + offset
                bits[global_bit_idx] = True
                if threshold == thresholds[-1]:
                    bit_info[global_bit_idx] = memberships[ring_size]
                else:
                    bit_info[global_bit_idx] = memberships[ring_size][:threshold]

    return bits, bit_info


def _pubchem_ring_count_template() -> tuple[dict[int, int], dict[int, tuple[tuple[int, ...], ...]]]:
    ring_sizes = (3, 4, 5, 6, 7, 8, 9, 10)
    return {ring_size: 0 for ring_size in ring_sizes}, {ring_size: () for ring_size in ring_sizes}


def _pubchem_func_1(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    counts, memberships = _pubchem_ring_count_template()
    for ring in mol.GetRingInfo().AtomRings():
        ring_size = len(ring)
        if ring_size in counts:
            counts[ring_size] += 1
            memberships[ring_size] += (tuple(ring),)
    return _pubchem_map_ring_counts(counts, memberships, bits, bit_info)


def _pubchem_atom_tuple_from_bond_ring(
    mol: Chem.Mol, bond_ring: tuple[int, ...]
) -> tuple[int, ...]:
    atom_indices: set[int] = set()
    for bond_idx in bond_ring:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_indices.add(bond.GetBeginAtomIdx())
        atom_indices.add(bond.GetEndAtomIdx())
    return tuple(sorted(atom_indices))


def _pubchem_ring_family_counts(
    mol: Chem.Mol, *, offset: int, accept_ring
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    bits = np.zeros(148, dtype=bool)
    bit_info: dict[int, tuple[tuple[int, ...], ...]] = {}
    counts, memberships = _pubchem_ring_count_template()
    for bond_ring in mol.GetRingInfo().BondRings():
        if accept_ring(bond_ring):
            ring_size = len(bond_ring)
            if ring_size in counts:
                counts[ring_size] += 1
                memberships[ring_size] += (_pubchem_atom_tuple_from_bond_ring(mol, bond_ring),)
    return _pubchem_map_ring_counts(counts, memberships, bits, bit_info, offset=offset)


def _pubchem_is_all_single(mol: Chem.Mol, bond_ring: tuple[int, ...]) -> bool:
    return all(
        mol.GetBondWithIdx(bond_idx).GetBondType().name == "SINGLE" for bond_idx in bond_ring
    )


def _pubchem_is_all_aromatic(mol: Chem.Mol, bond_ring: tuple[int, ...]) -> bool:
    return all(
        mol.GetBondWithIdx(bond_idx).GetBondType().name == "AROMATIC" for bond_idx in bond_ring
    )


def _pubchem_contains_atomic_number(
    mol: Chem.Mol, bond_ring: tuple[int, ...], atomic_number: int
) -> bool:
    for bond_idx in bond_ring:
        bond = mol.GetBondWithIdx(bond_idx)
        if (
            bond.GetBeginAtom().GetAtomicNum() == atomic_number
            or bond.GetEndAtom().GetAtomicNum() == atomic_number
        ):
            return True
    return False


def _pubchem_is_all_carbons(mol: Chem.Mol, bond_ring: tuple[int, ...]) -> bool:
    for bond_idx in bond_ring:
        bond = mol.GetBondWithIdx(bond_idx)
        if bond.GetBeginAtom().GetAtomicNum() != 6 or bond.GetEndAtom().GetAtomicNum() != 6:
            return False
    return True


def _pubchem_contains_heteroatom(mol: Chem.Mol, bond_ring: tuple[int, ...]) -> bool:
    for bond_idx in bond_ring:
        bond = mol.GetBondWithIdx(bond_idx)
        if bond.GetBeginAtom().GetAtomicNum() not in {1, 6}:
            return True
        if bond.GetEndAtom().GetAtomicNum() not in {1, 6}:
            return True
    return False


def _pubchem_is_unsaturated_nonaromatic(mol: Chem.Mol, bond_ring: tuple[int, ...]) -> bool:
    unsaturated = any(
        mol.GetBondWithIdx(bond_idx).GetBondType().name != "SINGLE" for bond_idx in bond_ring
    )
    nonaromatic = all(
        mol.GetBondWithIdx(bond_idx).GetBondType().name != "AROMATIC" for bond_idx in bond_ring
    )
    return unsaturated and nonaromatic


def _pubchem_func_2(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=1,
        accept_ring=lambda bond_ring: _pubchem_is_all_single(mol, bond_ring)
        or (_pubchem_is_all_aromatic(mol, bond_ring) and _pubchem_is_all_carbons(mol, bond_ring)),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_3(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=2,
        accept_ring=lambda bond_ring: _pubchem_is_all_single(mol, bond_ring)
        or (
            _pubchem_is_all_aromatic(mol, bond_ring)
            and _pubchem_contains_atomic_number(mol, bond_ring, 7)
        ),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_4(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=3,
        accept_ring=lambda bond_ring: _pubchem_is_all_single(mol, bond_ring)
        or (
            _pubchem_is_all_aromatic(mol, bond_ring)
            and _pubchem_contains_heteroatom(mol, bond_ring)
        ),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_5(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=4,
        accept_ring=lambda bond_ring: _pubchem_is_unsaturated_nonaromatic(mol, bond_ring)
        and _pubchem_is_all_carbons(mol, bond_ring),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_6(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=5,
        accept_ring=lambda bond_ring: _pubchem_is_unsaturated_nonaromatic(mol, bond_ring)
        and _pubchem_contains_atomic_number(mol, bond_ring, 7),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_7(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    family_bits, family_info = _pubchem_ring_family_counts(
        mol,
        offset=6,
        accept_ring=lambda bond_ring: _pubchem_is_unsaturated_nonaromatic(mol, bond_ring)
        and _pubchem_contains_heteroatom(mol, bond_ring),
    )
    bits |= family_bits
    bit_info.update(family_info)
    return bits, bit_info


def _pubchem_func_8(
    mol: Chem.Mol, bits: np.ndarray, bit_info: dict[int, tuple[tuple[int, ...], ...]]
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    aromatic_count = 0
    hetero_count = 0
    aromatic_memberships: tuple[tuple[int, ...], ...] = ()
    hetero_memberships: tuple[tuple[int, ...], ...] = ()

    for bond_ring in mol.GetRingInfo().BondRings():
        atom_tuple = _pubchem_atom_tuple_from_bond_ring(mol, bond_ring)
        if _pubchem_is_all_aromatic(mol, bond_ring):
            aromatic_count += 1
            aromatic_memberships += (atom_tuple,)
        if _pubchem_contains_heteroatom(mol, bond_ring):
            hetero_count += 1
            hetero_memberships += (atom_tuple,)

    for threshold, bit_idx in zip([1, 2, 3, 4], [140, 142, 144, 146], strict=True):
        if aromatic_count >= threshold:
            bits[bit_idx] = True
            bit_info[bit_idx] = (
                aromatic_memberships if threshold == 4 else aromatic_memberships[:threshold]
            )

    for threshold, bit_idx in zip([1, 2, 3, 4], [141, 143, 145, 147], strict=True):
        if hetero_count >= threshold and aromatic_count >= threshold:
            bits[bit_idx] = True
            bit_info[bit_idx] = (
                hetero_memberships if threshold == 4 else hetero_memberships[:threshold]
            )

    return bits, bit_info


def _pubchem_get_one_bit_info(mol: Chem.Mol, bit_id: int) -> tuple[tuple[int, ...], ...]:
    pattern, count = _pubchem_keys()[bit_id]
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) <= count:
        return ()
    return tuple(tuple(int(atom_idx) for atom_idx in match) for match in matches)


def _pubchem_part1(mol: Chem.Mol) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    bits = np.zeros(len(PUBCHEM_SMARTS_BIT_IDS), dtype=bool)
    bit_info: dict[int, tuple[tuple[int, ...], ...]] = {}
    for local_bit_index in range(len(PUBCHEM_SMARTS_BIT_IDS)):
        info = _pubchem_get_one_bit_info(mol, local_bit_index)
        if info:
            bits[local_bit_index] = True
            bit_info[local_bit_index] = info
    return bits, bit_info


def _pubchem_part2(mol: Chem.Mol) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    bits = np.zeros(148, dtype=bool)
    bit_info: dict[int, tuple[tuple[int, ...], ...]] = {}
    for func in (
        _pubchem_func_1,
        _pubchem_func_2,
        _pubchem_func_3,
        _pubchem_func_4,
        _pubchem_func_5,
        _pubchem_func_6,
        _pubchem_func_7,
        _pubchem_func_8,
    ):
        bits, bit_info = func(mol, bits, bit_info)
    return bits, bit_info


def pubchem_mol_to_fp_bit_info(
    mol: Chem.Mol,
) -> tuple[np.ndarray, dict[int, tuple[tuple[int, ...], ...]]]:
    """Return the full 881-bit PubChem activation vector and atom-set provenance."""
    fp_part1, bit_info_part1 = _pubchem_part1(mol)
    fp_part2, bit_info_part2 = _pubchem_part2(mol)

    fp = np.zeros(PUBCHEM_NBITS, dtype=bool)
    bit_info: dict[int, tuple[tuple[int, ...], ...]] = {}

    for local_idx, global_idx in enumerate(PUBCHEM_SMARTS_BIT_IDS):
        fp[global_idx] = bool(fp_part1[local_idx])
        if local_idx in bit_info_part1:
            bit_info[global_idx] = bit_info_part1[local_idx]

    for local_idx, is_on in enumerate(fp_part2.tolist(), start=115):
        fp[local_idx] = bool(is_on)
        if local_idx - 115 in bit_info_part2:
            bit_info[local_idx] = bit_info_part2[local_idx - 115]

    return fp, bit_info


def pubchem_bitinfos_to_atom_bond_indices(
    mol: Chem.Mol, bitinfos: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the atom and bond indices participating in one PubChem fingerprint bit."""
    atom_indices = tuple(sorted({int(atom_idx) for match in bitinfos for atom_idx in match}))
    bond_indices = _pubchem_atoms_to_bonds(mol, atom_indices)
    return atom_indices, bond_indices


def build_morgan_bit_memberships(
    mol: Chem.Mol,
    radius: int = 2,
    nbits: int = 1024,
    family_name: str = "morgan",
    family_index: int = 0,
    bit_offset: int = 0,
) -> tuple[FPPoolBitMembership, ...]:
    """Build active Morgan bit memberships with explicit atom-and-bond provenance.

    Folded Morgan bit collisions are unioned across contributing environments, matching the
    reference FPPool implementation.
    """
    bit_info: dict[int, tuple[tuple[int, int], ...]] = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=nbits, bitInfo=bit_info
    )

    memberships: list[FPPoolBitMembership] = []
    for family_bit_index, environments in sorted(bit_info.items()):
        atom_indices: set[int] = set()
        bond_indices: set[int] = set()
        for atom_idx, env_radius in environments:
            atom_indices.update(morgan_bit_environment_atom_indices(mol, atom_idx, env_radius))
            bond_indices.update(morgan_bit_environment_bond_indices(mol, atom_idx, env_radius))

        memberships.append(
            FPPoolBitMembership(
                family_name=family_name,
                family_index=family_index,
                family_bit_index=family_bit_index,
                global_bit_index=bit_offset + family_bit_index,
                atom_indices=tuple(sorted(atom_indices)),
                bond_indices=tuple(sorted(bond_indices)),
            )
        )

    return tuple(memberships)


def build_rdkit_bit_memberships(
    mol: Chem.Mol,
    nbits: int = 1024,
    min_path: int = 1,
    max_path: int = 5,
    family_name: str = "rdkit",
    family_index: int = 0,
    bit_offset: int = 0,
) -> tuple[FPPoolBitMembership, ...]:
    """Build active RDKit path-fingerprint memberships with explicit atom-and-bond provenance."""
    bit_info: dict[int, list[list[int]]] = {}
    RDKFingerprint(mol, fpSize=nbits, minPath=min_path, maxPath=max_path, bitInfo=bit_info)

    memberships: list[FPPoolBitMembership] = []
    for family_bit_index, bond_paths in sorted(bit_info.items()):
        atom_indices: set[int] = set()
        bond_indices: set[int] = set()
        for bond_path in bond_paths:
            path_atom_indices, path_bond_indices = rdkit_bit_path_atom_bond_indices(mol, bond_path)
            atom_indices.update(path_atom_indices)
            bond_indices.update(path_bond_indices)

        memberships.append(
            FPPoolBitMembership(
                family_name=family_name,
                family_index=family_index,
                family_bit_index=family_bit_index,
                global_bit_index=bit_offset + family_bit_index,
                atom_indices=tuple(sorted(atom_indices)),
                bond_indices=tuple(sorted(bond_indices)),
            )
        )

    return tuple(memberships)


def build_pubchem_bit_memberships(
    mol: Chem.Mol, family_name: str = "pubchem", family_index: int = 0, bit_offset: int = 0
) -> tuple[FPPoolBitMembership, ...]:
    """Build active PubChem fingerprint memberships with explicit atom-and-bond provenance."""
    _, bit_info = pubchem_mol_to_fp_bit_info(mol)

    memberships: list[FPPoolBitMembership] = []
    for family_bit_index, bitinfos in sorted(bit_info.items()):
        atom_indices, bond_indices = pubchem_bitinfos_to_atom_bond_indices(mol, bitinfos)
        memberships.append(
            FPPoolBitMembership(
                family_name=family_name,
                family_index=family_index,
                family_bit_index=family_bit_index,
                global_bit_index=bit_offset + family_bit_index,
                atom_indices=atom_indices,
                bond_indices=bond_indices,
            )
        )

    return tuple(memberships)


def build_morgan_atom_fp(mol: Chem.Mol, radius: int = 2, nbits: int = 1024) -> np.ndarray:
    """Build a dense atom-to-bit membership matrix from Morgan provenance."""
    atom_fp = np.zeros((mol.GetNumAtoms(), nbits), dtype=bool)
    for membership in build_morgan_bit_memberships(mol, radius=radius, nbits=nbits):
        atom_fp[list(membership.atom_indices), membership.family_bit_index] = True

    return atom_fp


def build_rdkit_atom_fp(
    mol: Chem.Mol, nbits: int = 1024, min_path: int = 1, max_path: int = 5
) -> np.ndarray:
    """Build a dense atom-to-bit membership matrix from RDKit path-fingerprint provenance."""
    atom_fp = np.zeros((mol.GetNumAtoms(), nbits), dtype=bool)
    for membership in build_rdkit_bit_memberships(
        mol, nbits=nbits, min_path=min_path, max_path=max_path
    ):
        atom_fp[list(membership.atom_indices), membership.family_bit_index] = True

    return atom_fp


def build_pubchem_atom_fp(mol: Chem.Mol) -> np.ndarray:
    """Build a dense atom-to-bit membership matrix from PubChem provenance."""
    atom_fp = np.zeros((mol.GetNumAtoms(), PUBCHEM_NBITS), dtype=bool)
    for membership in build_pubchem_bit_memberships(mol):
        atom_fp[list(membership.atom_indices), membership.family_bit_index] = True

    return atom_fp


def build_atoms_repr_atom_fp(mol: Chem.Mol) -> np.ndarray:
    """Build the synthetic atomic special family membership matrix."""
    return np.ones((mol.GetNumAtoms(), 1), dtype=bool)


def build_fppool_atom_fp(mol: Chem.Mol, config: FPPoolConfig) -> np.ndarray:
    """Build the concatenated FPPool atom-membership matrix for one molecule."""
    family_blocks: list[np.ndarray] = []
    if config.atoms_repr:
        family_blocks.append(build_atoms_repr_atom_fp(mol))

    for family_name in config.family_names:
        if family_name == "morgan":
            family_blocks.append(
                build_morgan_atom_fp(mol, radius=config.morgan_radius, nbits=config.morgan_nbits)
            )
        elif family_name == "rdkit":
            family_blocks.append(
                build_rdkit_atom_fp(
                    mol,
                    nbits=config.rdkit_nbits,
                    min_path=config.rdkit_min_path,
                    max_path=config.rdkit_max_path,
                )
            )
        elif family_name == "pubchem":
            family_blocks.append(build_pubchem_atom_fp(mol))
        else:
            raise ValueError(f"Unsupported FPPool family '{family_name}'.")

    return np.concatenate(family_blocks, axis=1)


def build_fppool_molecule_provenance(
    mol: Chem.Mol, config: FPPoolConfig
) -> FPPoolMoleculeProvenance:
    """Build explicit atom-and-bond provenance for the active FPPool families of one molecule."""
    bit_memberships: list[FPPoolBitMembership] = []
    family_index = 0
    bit_offset = 0

    if config.atoms_repr:
        bit_memberships.append(
            FPPoolBitMembership(
                family_name="atoms",
                family_index=family_index,
                family_bit_index=0,
                global_bit_index=bit_offset,
                atom_indices=tuple(range(mol.GetNumAtoms())),
                bond_indices=tuple(),
            )
        )
        family_index += 1
        bit_offset += 1

    for family_name in config.family_names:
        if family_name == "morgan":
            memberships = build_morgan_bit_memberships(
                mol,
                radius=config.morgan_radius,
                nbits=config.morgan_nbits,
                family_name=family_name,
                family_index=family_index,
                bit_offset=bit_offset,
            )
            bit_memberships.extend(memberships)
            bit_offset += config.morgan_nbits
            family_index += 1
        elif family_name == "rdkit":
            memberships = build_rdkit_bit_memberships(
                mol,
                nbits=config.rdkit_nbits,
                min_path=config.rdkit_min_path,
                max_path=config.rdkit_max_path,
                family_name=family_name,
                family_index=family_index,
                bit_offset=bit_offset,
            )
            bit_memberships.extend(memberships)
            bit_offset += config.rdkit_nbits
            family_index += 1
        elif family_name == "pubchem":
            memberships = build_pubchem_bit_memberships(
                mol, family_name=family_name, family_index=family_index, bit_offset=bit_offset
            )
            bit_memberships.extend(memberships)
            bit_offset += PUBCHEM_NBITS
            family_index += 1
        else:
            raise ValueError(f"Unsupported FPPool family '{family_name}'.")

    return FPPoolMoleculeProvenance(
        family_names=tuple(config.active_family_names),
        family_lengths=tuple(int(length) for length in config.family_lengths.tolist()),
        bit_memberships=tuple(bit_memberships),
    )


def load_or_create_fppool_atom_fps(
    mols: list[Chem.Mol],
    dataset_path: PathLike,
    config: FPPoolConfig,
    cache_root: PathLike | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load cached FPPool memberships or generate and persist them on cache miss."""
    cache_dir = derive_fppool_cache_dir(dataset_path, config, cache_root)
    meta_path = _meta_path(cache_dir)
    atom_fp_path = _atom_fp_path(cache_dir)

    if meta_path.exists() and atom_fp_path.exists():
        return load_fppool_cache(cache_dir, dataset_path, config)

    atom_fps = [build_fppool_atom_fp(mol, config) for mol in mols]
    save_fppool_cache(cache_dir, atom_fps, dataset_path, config)
    return atom_fps, config.family_lengths, config.active_family_names


def apply_fppool_metadata(
    datapoints: list[MoleculeDatapoint],
    dataset_path: PathLike,
    config: FPPoolConfig,
    cache_root: PathLike | None = None,
) -> None:
    """Attach cached or generated FPPool memberships to eager molecule datapoints."""
    mols = [dp.mol for dp in datapoints]
    atom_fps, family_lengths, family_names = load_or_create_fppool_atom_fps(
        mols, dataset_path, config, cache_root
    )

    if len(atom_fps) != len(datapoints):
        raise ValueError(
            "Number of cached/generated FPPool atom memberships must match datapoints."
        )

    for datapoint, atom_fp in zip(datapoints, atom_fps):
        if atom_fp.shape[0] != datapoint.mol.GetNumAtoms():
            raise ValueError(
                "FPPool atom memberships must contain one row per atom in the datapoint."
            )

        datapoint.atom_fp = atom_fp
        datapoint.fp_family_lengths = family_lengths.copy()
        datapoint.fp_family_names = list(family_names)
