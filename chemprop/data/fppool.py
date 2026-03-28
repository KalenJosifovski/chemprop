from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemprop.data.datapoints import MoleculeDatapoint


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
    atoms_repr : bool, default=False
        Reserved for future parity with the reference implementation's synthetic atom family.
    schema_version : str, default="1"
        A small cache-schema discriminator used as part of cache identity and metadata.
    """

    family_names: tuple[str, ...] = ("morgan",)
    morgan_nbits: int = 1024
    morgan_radius: int = 2
    atoms_repr: bool = False
    schema_version: str = "1"

    @property
    def family_lengths(self) -> np.ndarray:
        """The ordered family bit counts for the configured fingerprint families."""
        lengths = []
        for family_name in self.family_names:
            if family_name == "morgan":
                lengths.append(self.morgan_nbits)
            else:
                raise ValueError(f"Unsupported FPPool family '{family_name}'.")

        return np.asarray(lengths, dtype=int)

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["family_names"] = list(self.family_names)
        metadata["family_lengths"] = self.family_lengths.tolist()
        return metadata


def derive_fppool_cache_key(dataset_path: PathLike, config: FPPoolConfig) -> str:
    """Build a stable cache key from the dataset path and FPPool configuration."""
    payload = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "config": config.to_metadata(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def derive_fppool_cache_dir(
    dataset_path: PathLike,
    config: FPPoolConfig,
    cache_root: PathLike | None = None,
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
    cache_dir: PathLike,
    atom_fps: list[np.ndarray],
    dataset_path: PathLike,
    config: FPPoolConfig,
) -> None:
    """Save a directory-based FPPool cache for local single-process use."""
    cache_dir = Path(cache_dir)
    metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "family_names": list(config.family_names),
        "family_lengths": config.family_lengths.tolist(),
        "config": config.to_metadata(),
        "num_datapoints": len(atom_fps),
    }
    _write_json_atomic(_meta_path(cache_dir), metadata)
    _save_npz_atomic(_atom_fp_path(cache_dir), atom_fps)


def load_fppool_cache(
    cache_dir: PathLike,
    dataset_path: PathLike,
    config: FPPoolConfig,
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

    if family_names != list(config.family_names):
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


def build_morgan_atom_fp(mol: Chem.Mol, radius: int = 2, nbits: int = 1024) -> np.ndarray:
    """Build a dense atom-to-bit membership matrix from Morgan provenance."""
    bit_info: dict[int, tuple[tuple[int, int], ...]] = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=nbits,
        bitInfo=bit_info,
    )

    atom_fp = np.zeros((mol.GetNumAtoms(), nbits), dtype=bool)
    for bit_idx, environments in bit_info.items():
        for atom_idx, env_radius in environments:
            atom_indices = morgan_bit_environment_atom_indices(mol, atom_idx, env_radius)
            # Folded Morgan bits can map multiple environments onto the same column. The reference
            # FPPool implementation unions those memberships, so we do the same here.
            atom_fp[list(atom_indices), bit_idx] = True

    return atom_fp


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

    atom_fps = [
        build_morgan_atom_fp(mol, radius=config.morgan_radius, nbits=config.morgan_nbits)
        for mol in mols
    ]
    save_fppool_cache(cache_dir, atom_fps, dataset_path, config)
    return atom_fps, config.family_lengths, list(config.family_names)


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
        raise ValueError("Number of cached/generated FPPool atom memberships must match datapoints.")

    for datapoint, atom_fp in zip(datapoints, atom_fps):
        if atom_fp.shape[0] != datapoint.mol.GetNumAtoms():
            raise ValueError("FPPool atom memberships must contain one row per atom in the datapoint.")

        datapoint.atom_fp = atom_fp
        datapoint.fp_family_lengths = family_lengths.copy()
        datapoint.fp_family_names = list(family_names)
