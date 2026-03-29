"""Integration coverage for molecule-only FPPool training."""

from pathlib import Path
from unittest.mock import patch

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import models, nn
from chemprop.cli.utils import build_data_from_files
from chemprop.data import FPPoolConfig, MoleculeDataset, collate_batch, derive_fppool_cache_dir


@pytest.fixture
def fppool_data_path(data_dir: Path) -> Path:
    return data_dir / "regression" / "mol" / "mol.csv"


@pytest.fixture
def fppool_dataloader(fppool_data_path: Path, tmp_path: Path) -> DataLoader:
    config = FPPoolConfig(morgan_nbits=64, morgan_radius=2)
    (molecule_data,) = build_data_from_files(
        fppool_data_path,
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        target_cols=["lipo"],
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

    dset = MoleculeDataset(molecule_data)
    dset.normalize_targets()
    return DataLoader(dset, batch_size=32, collate_fn=collate_batch)


@pytest.mark.integration
def test_fppool_quick(fppool_dataloader: DataLoader):
    mpnn = models.MPNN(
        nn.BondMessagePassing(), nn.FPPoolAggregation(), nn.RegressionFFN(), batch_norm=True
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, fppool_dataloader, None)


@pytest.mark.integration
def test_fppool_overfit(fppool_dataloader: DataLoader):
    mpnn = models.MPNN(
        nn.BondMessagePassing(), nn.FPPoolAggregation(), nn.RegressionFFN(), batch_norm=True
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=50,
        overfit_batches=1.0,
    )
    trainer.fit(mpnn, fppool_dataloader)

    errors = []
    for batch in fppool_dataloader:
        bmg, _, _, targets, *_, fppool_batch = batch
        preds = mpnn(bmg, fppool_batch=fppool_batch)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05


@pytest.mark.integration
def test_fppool_cached_data_prep_path_uses_cache(fppool_data_path: Path, tmp_path: Path):
    config = FPPoolConfig(morgan_nbits=64, morgan_radius=2)
    cache_root = tmp_path / ".cache"

    (first_data,) = build_data_from_files(
        fppool_data_path,
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        target_cols=["lipo"],
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
        fppool_cache_root=cache_root,
        n_workers=0,
    )

    cache_dir = derive_fppool_cache_dir(fppool_data_path, config, cache_root=cache_root)
    assert cache_dir.exists()
    assert cache_dir.joinpath("meta.json").exists()
    assert cache_dir.joinpath("atom_fp.npz").exists()
    assert all(dp.atom_fp is not None for dp in first_data)

    with patch(
        "chemprop.data.fppool.build_morgan_atom_fp",
        side_effect=AssertionError("Expected the second pass to consume the cache."),
    ):
        (second_data,) = build_data_from_files(
            fppool_data_path,
            no_header_row=False,
            smiles_cols=["smiles"],
            rxn_cols=None,
            target_cols=["lipo"],
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
            fppool_cache_root=cache_root,
            n_workers=0,
        )

    for first_dp, second_dp in zip(first_data, second_data):
        assert first_dp.atom_fp is not None
        assert second_dp.atom_fp is not None
        assert first_dp.atom_fp.shape == second_dp.atom_fp.shape
        assert (first_dp.atom_fp == second_dp.atom_fp).all()


@pytest.mark.integration
def test_fppool_multifamily_quick(fppool_data_path: Path, tmp_path: Path):
    config = FPPoolConfig(
        family_names=("morgan", "rdkit", "pubchem"),
        morgan_nbits=64,
        rdkit_nbits=64,
        atoms_repr=True,
    )
    (molecule_data,) = build_data_from_files(
        fppool_data_path,
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        target_cols=["lipo"],
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

    dset = MoleculeDataset(molecule_data)
    dset.normalize_targets()
    dataloader = DataLoader(dset, batch_size=16, collate_fn=collate_batch)

    mpnn = models.MPNN(
        nn.BondMessagePassing(), nn.FPPoolAggregation(), nn.RegressionFFN(), batch_norm=True
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, dataloader, None)
