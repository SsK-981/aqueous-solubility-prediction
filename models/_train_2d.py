"""Fit fresh structure-only MDM, GNN and SMI checkpoints on public clean data."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch_geometric.data import Batch

from .forest import BASELINE, baseline_features
from .gnn.model import GraphModel, encode_graph
from .mdm.model import DescriptorModel
from .pretrained import _state_hash
from .smi.model import SequenceModel, build_vocabulary, encode_sequences

SMI_MAX_LENGTH = 256


def _descriptor_matrix(smiles):
    rows = []
    for structure in smiles:
        mol = Chem.MolFromSmiles(structure)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty molecular SMILES: {structure!r}")
        features = baseline_features(mol)
        rows.append([features[name] for name in BASELINE])
    return np.asarray(rows, dtype=np.float64)


def fit(family, train_rows, validation_rows, checkpoint_root, *, epochs, seed):
    """Fit on training rows and save the epoch with the lowest validation MSE."""
    if family not in {"mdm", "gnn", "smi"}:
        raise ValueError(f"Unsupported 2D model family: {family!r}")
    if not train_rows or not validation_rows or epochs < 1:
        raise ValueError("Training and validation rows and at least one epoch are required")

    folder = Path(checkpoint_root) / family
    if folder.exists():
        raise FileExistsError(f"Checkpoint folder already exists: {folder}")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    rows = [*train_rows, *validation_rows]
    smiles = [row["canonical_smiles"] for row in rows]
    target = torch.tensor([float(row["observed_logS"]) for row in rows], dtype=torch.float32)
    if not torch.isfinite(target).all():
        raise ValueError("Expected finite observed_logS values")
    train_count = len(train_rows)

    # The shared checkpoint reader requires descriptor metadata for each family.
    # Only MDM consumes the resulting descriptor features in its forward pass.
    train_descriptors = _descriptor_matrix(smiles[:train_count])
    scaler = StandardScaler().fit(train_descriptors)
    preprocessing = {
        "n_features": len(BASELINE),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    metadata = {
        "family": family.upper(),
        "temperature": False,
        "target": "observed_logS",
        "target_units": "log10(mol/L)",
        "fit": "fresh_public_training",
        "preprocessing": preprocessing,
    }

    batch_size = 64
    if family == "mdm":
        model = DescriptorModel(len(BASELINE))
        descriptors = np.concatenate(
            (train_descriptors, _descriptor_matrix(smiles[train_count:]))
        )
        features = torch.from_numpy(
            scaler.transform(descriptors).astype(np.float32)
        )
        batch_features = features.__getitem__
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)
    elif family == "gnn":
        model = GraphModel()
        graphs = {structure: encode_graph(structure) for structure in dict.fromkeys(smiles)}

        def batch_features(indices):
            return Batch.from_data_list([graphs[smiles[index]] for index in indices.tolist()])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        vocabulary = build_vocabulary(smiles[:train_count])
        tokens = encode_sequences(smiles, vocabulary, SMI_MAX_LENGTH)
        model = SequenceModel(len(vocabulary) + 2)
        batch_size = 16
        batch_features = torch.from_numpy(tokens).__getitem__
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        metadata["vocabulary"] = vocabulary
        metadata["sequence_encoding"] = {"maxlen": SMI_MAX_LENGTH, "truncate": False}

    validation_indices = torch.arange(train_count, len(rows)).split(batch_size)
    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(train_count, generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch_features(indices)), target[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{family.upper()} training loss is non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_mse = sum(
                F.mse_loss(model(batch_features(indices)), target[indices], reduction="sum").item()
                for indices in validation_indices
            ) / len(validation_rows)
        if not np.isfinite(validation_mse):
            raise RuntimeError(f"{family.upper()} validation loss is non-finite at epoch {epoch}")
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    metadata["training"] = {
        "seed": seed,
        "epochs_completed": epochs,
        "training_rows": train_count,
        "validation_rows": len(validation_rows),
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
    }
    metadata["model_state_sha256"] = _state_hash(best_state)

    # Publish the pair together only after both writes have completed.
    with TemporaryDirectory(dir=folder.parent) as work:
        staged = Path(work) / family
        staged.mkdir()
        torch.save(best_state, staged / "model.pt")
        (staged / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.rename(folder)
    return folder
