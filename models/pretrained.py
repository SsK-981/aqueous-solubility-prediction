"""Inference loaders for the fitted structure-only SOMAS neural models.

Checkpoint layout: ``<checkpoint_root>/<family>/model.pt`` and
``<checkpoint_root>/<family>/metadata.json`` where family is mdm, gnn, or smi.
The returned callable accepts a canonical SMILES string and ``None`` for
temperature, and returns predicted log10(mol/L). These are the thesis's fitted
adaptations, not the original authors' pretrained weights.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .forest import BASELINE, baseline_features
from .gnn.model import GraphModel, encode_graph
from .mdm.model import DescriptorModel
from .smi.model import SequenceModel, encode_sequences


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_pretrained(family: str, checkpoint_root: str | Path):
    """Load a fitted MDM/GNN/SMI model and return ``predict(smiles, None)``.

    Parameters
    ----------
    family:
        One of ``"mdm"``, ``"gnn"``, or ``"smi"``.
    checkpoint_root:
        Directory containing one subdirectory per family (e.g. ``checkpoints/neural``).

    The model needs the same descriptors, scaler and (for SMI) token vocabulary
    used during thesis training. Those values are bundled in metadata.json.
    These selected fits are structure-only and therefore require temperature=None.
    """
    family = family.lower()
    if family not in {"mdm", "gnn", "smi"}:
        raise ValueError("family must be one of: mdm, gnn, smi")
    checkpoint_root = Path(checkpoint_root)
    # The archive has mdm/, gnn/, smi/ at its root and is extracted into the
    # repository's checkpoints/ folder. Also accept checkpoints/neural/.
    folder = checkpoint_root / family
    if not (folder / "metadata.json").is_file() and (checkpoint_root / "neural" / family / "metadata.json").is_file():
        folder = checkpoint_root / "neural" / family
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata.get("family") != family.upper() or metadata.get("temperature") is not False:
        raise ValueError(f"Checkpoint metadata does not identify a structure-only {family.upper()} model")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    expected = metadata.get("model_state_sha256")
    if not expected or _state_hash(checkpoint) != expected:
        raise ValueError(f"Model checkpoint integrity check failed for {family.upper()}")

    prep = metadata["preprocessing"]
    # The structure-only benchmark stores these directly; temperature studies
    # nest them under "descriptor". Keep the reader tolerant of both schemas.
    desc_prep = prep.get("descriptor", prep)
    if desc_prep.get("n_features") != len(BASELINE):
        raise ValueError("Unexpected descriptor count in neural checkpoint")
    desc_mean = np.asarray(desc_prep["scaler_mean"], dtype=np.float64)
    desc_scale = np.asarray(desc_prep["scaler_scale"], dtype=np.float64)
    if desc_mean.shape != (8,) or desc_scale.shape != (8,):
        raise ValueError("Invalid feature scaling metadata")
    if np.any(desc_scale == 0):
        raise ValueError("Invalid zero feature scale in checkpoint metadata")

    if family == "mdm":
        model = DescriptorModel(8)
    elif family == "gnn":
        model = GraphModel()
    else:
        vocab = metadata.get("vocabulary")
        encoding = metadata.get("sequence_encoding") or {}
        if not isinstance(vocab, dict) or not encoding.get("maxlen"):
            raise ValueError("SMI checkpoint is missing its fitted vocabulary or sequence length")
        model = SequenceModel(len(vocab) + 2)
        maxlen = int(encoding["maxlen"])

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    torch.set_num_threads(1)

    @lru_cache(maxsize=4096)
    def descriptors(smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Invalid or empty SMILES")
        values = baseline_features(mol)
        return np.asarray([values[name] for name in BASELINE], dtype=np.float64)

    def predict(canonical_smiles: str, temperature_kelvin: float | None = None) -> float:
        if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
            raise ValueError("canonical_smiles must be a nonempty string")
        if temperature_kelvin is not None:
            raise ValueError("This fitted model is structure-only; omit temperature")
        raw = descriptors(canonical_smiles)
        x = ((raw - desc_mean) / desc_scale).astype(np.float32)
        with torch.inference_mode():
            if family == "mdm":
                inputs = torch.from_numpy(x[None, :])
                output = model(inputs)
            elif family == "gnn":
                graph = encode_graph(canonical_smiles)
                output = model(graph)
            else:
                tokens = encode_sequences([canonical_smiles], vocab, maxlen)
                output = model(torch.as_tensor(tokens, dtype=torch.long))
        result = float(output.reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError("Model returned a non-finite prediction")
        return result

    return predict
