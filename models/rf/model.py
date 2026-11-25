"""Descriptor and Morgan-fingerprint inputs for the fitted Random Forest."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

BASELINE = [
    "MolWt", "MolLogP", "TPSA", "RotatableBonds", "HBD", "HBA",
    "AromaticRings", "FractionCSP3",
]


def baseline_features(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


TEMP = ["Temperature_K", "InvTemperature_1000_per_K"]


FP_BITS = 2048


FP_RADIUS = 2


def _morgan(smiles: pd.Series) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    values = np.zeros((len(smiles), FP_BITS), dtype=np.float32)
    for i, smi in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES at input row {i}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), values[i])
    names = [f"Morgan_r{FP_RADIUS}_{i:04d}" for i in range(FP_BITS)]
    return values, names


def forest_prediction(bundle, features, temperature, smiles):
    desc = bundle['scalers']['baseline'].transform(
        np.array([[features[name] for name in BASELINE]], dtype=float))
    temp = bundle['scalers']['temperature'].transform(
        np.array([[temperature, 1000 / temperature]], dtype=float))
    fingerprint, names = _morgan(pd.Series([smiles]))
    if bundle['feature_names'] != [*BASELINE, *names, *TEMP]:
        raise ValueError('Unexpected Random Forest feature order.')
    vector = np.concatenate([desc, fingerprint, temp], axis=1).astype(np.float32)
    return float(bundle['model'].predict(vector.astype(np.float64))[0])
