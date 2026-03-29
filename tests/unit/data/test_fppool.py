import numpy as np
from pandas import DataFrame

from chemprop.cli.utils import build_data_from_files
from chemprop.data import (
    FPPoolConfig,
    build_atoms_repr_atom_fp,
    build_fppool_atom_fp,
    build_morgan_atom_fp,
    derive_fppool_cache_dir,
    derive_fppool_cache_key,
    load_or_create_fppool_atom_fps,
)
from chemprop.utils import make_mol


def test_build_morgan_atom_fp_shape_and_active_bits():
    mol = make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)

    atom_fp = build_morgan_atom_fp(mol, radius=2, nbits=32)
    atom_fp_repeat = build_morgan_atom_fp(mol, radius=2, nbits=32)

    assert atom_fp.dtype == bool
    assert atom_fp.shape == (mol.GetNumAtoms(), 32)
    assert np.array_equal(atom_fp, atom_fp_repeat)

    active_columns = np.where(atom_fp.any(axis=0))[0]
    inactive_columns = np.where(~atom_fp.any(axis=0))[0]

    assert len(active_columns) > 0
    assert np.all(atom_fp[:, active_columns].any(axis=0))
    assert np.all(~atom_fp[:, inactive_columns].any(axis=0))


def test_build_atoms_repr_atom_fp_is_single_all_atom_column():
    mol = make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)

    atom_fp = build_atoms_repr_atom_fp(mol)

    assert atom_fp.dtype == bool
    assert atom_fp.shape == (mol.GetNumAtoms(), 1)
    assert np.all(atom_fp)


def test_build_fppool_atom_fp_prepends_atoms_repr_family():
    mol = make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=True)

    atom_fp = build_fppool_atom_fp(mol, config)

    assert atom_fp.dtype == bool
    assert atom_fp.shape == (mol.GetNumAtoms(), 33)
    assert np.all(atom_fp[:, 0])
    assert np.array_equal(
        atom_fp[:, 1:],
        build_morgan_atom_fp(mol, radius=config.morgan_radius, nbits=config.morgan_nbits),
    )


def test_load_or_create_fppool_atom_fps_round_trips_cache(tmp_path):
    mols = [
        make_mol("CCO", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False),
        make_mol("CCN", keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False),
    ]
    dataset_path = tmp_path / "molecules.csv"
    dataset_path.write_text("smiles,y\nCCO,0\nCCN,1\n", encoding="utf-8")
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=True)

    atom_fps_1, family_lengths_1, family_names_1 = load_or_create_fppool_atom_fps(
        mols, dataset_path, config, cache_root=tmp_path / ".cache"
    )
    atom_fps_2, family_lengths_2, family_names_2 = load_or_create_fppool_atom_fps(
        mols, dataset_path, config, cache_root=tmp_path / ".cache"
    )

    cache_dir = derive_fppool_cache_dir(dataset_path, config, cache_root=tmp_path / ".cache")
    assert cache_dir.joinpath("meta.json").exists()
    assert cache_dir.joinpath("atom_fp.npz").exists()
    assert len(atom_fps_1) == len(mols)
    assert family_names_1 == ["atoms", "morgan"]
    assert np.array_equal(family_lengths_1, np.array([1, 32]))
    assert family_names_1 == family_names_2
    assert np.array_equal(family_lengths_1, family_lengths_2)
    for atom_fp_1, atom_fp_2, mol in zip(atom_fps_1, atom_fps_2, mols):
        assert atom_fp_1.dtype == bool
        assert atom_fp_1.shape == (mol.GetNumAtoms(), 33)
        assert np.array_equal(atom_fp_1, atom_fp_2)


def test_fppool_cache_key_changes_with_path_and_config(tmp_path):
    path_1 = tmp_path / "a.csv"
    path_2 = tmp_path / "b.csv"
    path_1.write_text("smiles,y\nCCO,0\n", encoding="utf-8")
    path_2.write_text("smiles,y\nCCO,0\n", encoding="utf-8")

    key_1 = derive_fppool_cache_key(path_1, FPPoolConfig(morgan_nbits=32, morgan_radius=2))
    key_2 = derive_fppool_cache_key(path_2, FPPoolConfig(morgan_nbits=32, morgan_radius=2))
    key_3 = derive_fppool_cache_key(path_1, FPPoolConfig(morgan_nbits=64, morgan_radius=2))
    key_4 = derive_fppool_cache_key(
        path_1, FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=False)
    )

    assert key_1 != key_2
    assert key_1 != key_3
    assert key_1 != key_4


def test_build_data_from_files_attaches_fppool_metadata_with_reordered_atoms(tmp_path):
    data_path = tmp_path / "data.csv"
    DataFrame({"smiles": ["[CH3:2][OH:1]"], "y": [1.0]}).to_csv(data_path, index=False)
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=True)

    data = build_data_from_files(
        data_path,
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        target_cols=["y"],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        descriptor_cols=None,
        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=True,
        use_cuikmolmaker_featurization=False,
        fppool_config=config,
        fppool_cache_root=tmp_path / ".cache",
        n_workers=0,
    )

    datapoint = data[0][0]
    expected_atom_fp = build_fppool_atom_fp(datapoint.mol, config)

    assert datapoint.atom_fp is not None
    assert datapoint.atom_fp.dtype == bool
    assert datapoint.atom_fp.shape[0] == datapoint.mol.GetNumAtoms()
    assert np.array_equal(datapoint.atom_fp, expected_atom_fp)
    assert np.array_equal(datapoint.fp_family_lengths, np.array([1, 32]))
    assert datapoint.fp_family_names == ["atoms", "morgan"]


def test_build_data_from_files_preserves_morgan_only_behavior_when_atoms_repr_disabled(tmp_path):
    data_path = tmp_path / "data.csv"
    DataFrame({"smiles": ["CCO"], "y": [1.0]}).to_csv(data_path, index=False)
    config = FPPoolConfig(morgan_nbits=32, morgan_radius=2, atoms_repr=False)

    data = build_data_from_files(
        data_path,
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        target_cols=["y"],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        descriptor_cols=None,
        molecule_featurizers=None,
        keep_h=False,
        add_h=False,
        ignore_stereo=False,
        reorder_atoms=False,
        use_cuikmolmaker_featurization=False,
        fppool_config=config,
        fppool_cache_root=tmp_path / ".cache",
        n_workers=0,
    )

    datapoint = data[0][0]
    expected_atom_fp = build_morgan_atom_fp(
        datapoint.mol, radius=config.morgan_radius, nbits=config.morgan_nbits
    )

    assert np.array_equal(datapoint.atom_fp, expected_atom_fp)
    assert np.array_equal(datapoint.fp_family_lengths, np.array([32]))
    assert datapoint.fp_family_names == ["morgan"]
