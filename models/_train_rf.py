"""Fit the Random Forest with the same feature layout as its predictor."""

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .forest import BASELINE, TEMP, _morgan, baseline_features


def _inputs(rows):
    smiles = [row["canonical_smiles"] for row in rows]
    descriptors = [baseline_features(Chem.MolFromSmiles(value)) for value in smiles]
    descriptor_matrix = np.array([[row[name] for name in BASELINE] for row in descriptors])
    temperature = np.array([float(row["temperature_K"]) for row in rows])
    fingerprint, names = _morgan(pd.Series(smiles))
    return descriptor_matrix, np.column_stack([temperature, 1000 / temperature]), fingerprint, names


def fit(family, train_rows, validation_rows, checkpoint_root, *, epochs, seed):
    desc, temp, fingerprints, names = _inputs(train_rows)
    scalers = {"baseline": StandardScaler().fit(desc), "temperature": StandardScaler().fit(temp)}

    def assemble(desc, temp, fingerprints):
        return np.column_stack([
            scalers["baseline"].transform(desc), fingerprints, scalers["temperature"].transform(temp),
        ]).astype(np.float32)

    model = RandomForestRegressor(n_estimators=300, random_state=seed)
    model.fit(assemble(desc, temp, fingerprints), [float(row["observed_logS"]) for row in train_rows])
    desc, temp, fingerprints, _ = _inputs(validation_rows)
    prediction = model.predict(assemble(desc, temp, fingerprints))
    actual = np.array([float(row["observed_logS"]) for row in validation_rows])
    print(f"Validation MAE: {np.mean(np.abs(prediction - actual)):.4f} log units.")
    folder = checkpoint_root / family
    folder.mkdir()
    joblib.dump({
        "model": model, "scalers": scalers, "feature_names": [*BASELINE, *names, *TEMP],
    }, folder / f"seed_{seed}.joblib", compress=3)
    return folder
