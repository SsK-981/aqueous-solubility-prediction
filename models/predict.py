"""Predict one solubility or score a cleaned test CSV from the repository root."""
import argparse
from functools import cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from data_preparation.clean import prepare, read_rows, write_rows

SEEDS = (42, 101, 202)
STRUCTURE_MODELS = ("esol", "mdm", "gnn", "smi")
TEMPERATURE_MODELS = ("rf", "sch", "schnet")
OUTPUT_FIELDS = (
    "prediction_status", "prediction_issue", "predicted_logS",
    "predicted_mol_L", "absolute_logS_error",
)


def _checkpoint_seeds(directory):
    fit_record = directory / "fit.json"
    if fit_record.is_file():
        return (int(json.loads(fit_record.read_text())["seed"]),)
    return SEEDS


def load_predictor(family, checkpoint_root):
    if family == "esol":
        from .esol import esol_log_s

        return lambda smiles, temperature: esol_log_s(smiles)

    if family in ("mdm", "gnn", "smi"):
        from .pretrained import load_pretrained

        return load_pretrained(family, checkpoint_root)

    import numpy as np
    from rdkit import Chem
    from .forest import baseline_features

    if family == "rf":
        import joblib
        from .forest import forest_prediction

        directory = checkpoint_root / family
        bundles = [joblib.load(directory / f"seed_{seed}.joblib") for seed in _checkpoint_seeds(directory)]

        @cache
        def features_for(smiles):
            return baseline_features(Chem.MolFromSmiles(smiles))

        def random_forest(smiles, temperature):
            features = features_for(smiles)
            return mean(forest_prediction(bundle, features, temperature, smiles) for bundle in bundles)

        return random_forest

    import torch
    from .sch.conformers import SEEDS as GEOMETRY_SEEDS, generate_conformer
    from .sch.model import (
        DESCRIPTORS, CoordinateRegressor, JointReadout, SchNetWithDescriptors,
    )

    torch.set_num_threads(1)
    custom = family == "schnet"
    ensemble = []
    directory = checkpoint_root / family
    for seed in _checkpoint_seeds(directory):
        stem = directory / f"seed_{seed}"
        checkpoint = stem.with_suffix(".pt")
        metadata = json.loads(stem.with_suffix(".json").read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata.get("checkpoint_sha256"):
            raise ValueError(f"SchNet checkpoint verification failed for seed {seed}.")
        model = (JointReadout(SchNetWithDescriptors(len(DESCRIPTORS)))
                 if custom else CoordinateRegressor())
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        ensemble.append((model, metadata))

    @cache
    def geometry_for(smiles):
        geometries = [generate_conformer(smiles, seed) for seed in GEOMETRY_SEEDS]
        for geometry in geometries:
            if geometry["status"] != "evaluated":
                raise ValueError(f"SchNet geometry unavailable ({geometry['status']}).")
        if len({geometry["pairs"] for geometry in geometries}) != 1:
            raise ValueError("Conformer runs disagreed on donor–acceptor pair count.")
        geometry = min(geometries, key=lambda entry: entry["energy"])
        atoms = torch.tensor(geometry["atomic_numbers"], dtype=torch.long)[None]
        positions = torch.tensor(geometry["coordinates"], dtype=torch.float32)[None]
        descriptors = None
        if custom:
            features = baseline_features(Chem.MolFromSmiles(smiles))
            descriptors = np.array([features[name] for name in DESCRIPTORS])
        return atoms, positions, geometry["pairs"], descriptors

    def schnet(smiles, temperature):
        atoms, positions, pairs, descriptors = geometry_for(smiles)
        values = []
        for model, metadata in ensemble:
            context = (np.array([temperature, 1000 / temperature, pairs, 0])
                       - metadata["global_mean"]) / metadata["global_scale"]
            context[3] = 0
            inputs = (atoms, positions, atoms != 0, torch.tensor(context, dtype=torch.float32)[None])
            if custom:
                scaled = (descriptors - metadata["descriptor_mean"]) / metadata["descriptor_scale"]
                inputs += (torch.tensor(scaled, dtype=torch.float32)[None],)
            with torch.inference_mode():
                prediction = model(*inputs)
            values.append(float(prediction[0]))
        return mean(values)

    return schnet


def score_rows(predictor, rows, *, uses_temperature=True):
    for row in rows:
        result = dict.fromkeys(OUTPUT_FIELDS, "")
        structure_only = row["input_status"] == "structure_only"
        if row["input_status"] != "ready" and not (structure_only and not uses_temperature):
            result.update(prediction_status="review_input", prediction_issue=row["input_issue"])
        else:
            try:
                temperature = float(row["temperature_K"]) if uses_temperature else None
                if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
                    raise ValueError("Temperature must be finite and above 0 K.")
                value = predictor(row["canonical_smiles"], temperature)
                if not math.isfinite(value):
                    raise ValueError("Model returned a non-finite estimate.")
                result.update(prediction_status="scored", predicted_logS=value,
                              predicted_mol_L=10 ** value)
            except (ValueError, OverflowError) as error:
                result.update(prediction_status="unsupported", prediction_issue=str(error))
            else:
                if row["observed_logS"] != "":
                    result["absolute_logS_error"] = abs(value - float(row["observed_logS"]))
        yield {**row, **result}


def main(family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if family is None:
        parser.add_argument("--model", choices=STRUCTURE_MODELS + TEMPERATURE_MODELS, default="esol")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"),
                        help="Directory containing the extracted model bundles.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles")
    inputs.add_argument("--input", type=Path, help="CSV produced by data_preparation.clean.")
    parser.add_argument("--celsius", type=float)
    parser.add_argument("--output", type=Path, help="New path for batch results.")
    args = parser.parse_args()
    if args.input is not None and (args.output is None or args.celsius is not None):
        parser.error("Batch prediction requires --output and uses temperatures already in the CSV.")
    family = family or args.model
    uses_temperature = family in TEMPERATURE_MODELS
    if args.smiles is not None and (args.output is not None or (uses_temperature and args.celsius is None)):
        parser.error("Single-molecule prediction prints the result. RF and both SchNet variants require --celsius.")
    try:
        predictor = load_predictor(family, args.checkpoints)
        if args.smiles is not None:
            row = {"smiles": args.smiles, "temperature": args.celsius, "temperature_unit": "C"}
            result = next(score_rows(predictor, prepare([row]), uses_temperature=uses_temperature))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["prediction_status"] == "scored" else 2
        headers, rows = read_rows(args.input, cleaned=True)
        if set(OUTPUT_FIELDS).intersection(headers):
            parser.error("Use an unscored CSV created by python -m data_preparation.clean.")
        scored = list(score_rows(predictor, rows, uses_temperature=uses_temperature))
        write_rows(args.output, [*headers, *OUTPUT_FIELDS], scored)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Prediction unavailable: {error}\n")
    count = sum(row["prediction_status"] == "scored" for row in scored)
    print(f"Scored {count} of {len(scored)} rows. All input rows are retained.")
    errors = [row["absolute_logS_error"] for row in scored if row["absolute_logS_error"] != ""]
    if errors:
        print(f"Measurement MAE on {len(errors)} scored rows with observations: {mean(errors):.4f} log units.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
