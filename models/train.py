"""Train a model on measured solubilities from a prepared CSV."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

from data_preparation.clean import read_rows
from .predict import STRUCTURE_MODELS, TEMPERATURE_MODELS

TRAINABLE_MODELS = STRUCTURE_MODELS[1:] + TEMPERATURE_MODELS


def measured_rows(path, uses_temperature):
    _, rows = read_rows(path, cleaned=True)
    eligible = []
    for row in rows:
        allowed = row["input_status"] == "ready" or (
            not uses_temperature and row["input_status"] == "structure_only"
        )
        if not allowed or not row["observed_logS"]:
            continue
        if not math.isfinite(float(row["observed_logS"])):
            raise ValueError("Measured logS must be finite.")
        if uses_temperature:
            temperature = float(row["temperature_K"])
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Temperature must be finite and above 0 K.")
        eligible.append(row)
    if not eligible:
        raise ValueError(f"No eligible measured rows in {path}.")
    return eligible, len(rows) - len(eligible)


def identity_groups(rows):
    keys = {}
    for row in rows:
        smiles = row["canonical_smiles"]
        if smiles not in keys:
            molecule = Chem.MolFromSmiles(smiles)
            if not molecule:
                raise ValueError(f"Invalid canonical SMILES: {smiles!r}.")
            key = Chem.MolToInchiKey(molecule)
            keys[smiles] = key.split("-")[0] if key else smiles
    return [keys[row["canonical_smiles"]] for row in rows]


def split_rows(rows, validation_path, uses_temperature, seed):
    groups = identity_groups(rows)
    if validation_path:
        validation, skipped = measured_rows(validation_path, uses_temperature)
        if set(groups).intersection(identity_groups(validation)):
            raise ValueError("Training and validation contain overlapping compound identities.")
        return rows, validation, skipped
    if len(set(groups)) < 2:
        raise ValueError("At least two distinct compound identities are needed.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, validation_idx = next(split.split(rows, groups=groups))
    return [rows[i] for i in train_idx], [rows[i] for i in validation_idx], 0


def main(family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if family is None:
        parser.add_argument("--model", choices=TRAINABLE_MODELS, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Prepared CSV with measured solubilities.")
    parser.add_argument("--validation", type=Path, help="Optional prepared CSV with distinct validation compounds.")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/new-fits"))
    parser.add_argument("--epochs", type=int, default=20, help="Maximum neural training epochs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    family = family or args.model
    if args.epochs < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("Use a positive epoch count and a seed from 0 to 4294967295.")
    uses_temperature = family in TEMPERATURE_MODELS
    destination = args.checkpoints / family
    try:
        if destination.exists():
            raise ValueError(f"Choose another --checkpoints directory. {destination} already exists.")
        rows, skipped = measured_rows(args.input, uses_temperature)
        train_rows, validation_rows, skipped_validation = split_rows(
            rows, args.validation, uses_temperature, args.seed
        )
        print(f"Prepared {len(train_rows)} training and {len(validation_rows)} validation rows. "
              f"Skipped {skipped + skipped_validation} ineligible input rows.")
        if family == "rf":
            from ._train_rf import fit
        elif family in ("sch", "schnet"):
            from ._train_3d import fit
        else:
            from ._train_2d import fit
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".fit-", dir=args.checkpoints) as staging:
            staging = Path(staging)
            report = fit(family, train_rows, validation_rows, staging, epochs=args.epochs, seed=args.seed)
            if family in ("sch", "schnet"):
                print(f"Usable geometry for {report['train']['used']} training and "
                      f"{report['validation']['used']} validation rows. "
                      f"Best validation MAE: {report['validation_mae']:.4f} log units.")
            folder = staging / family
            (folder / "fit.json").write_text(json.dumps({"seed": args.seed}, indent=2) + "\n")
            folder.rename(destination)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Training unavailable: {error}\n")
    print(f"Saved checkpoint to {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
