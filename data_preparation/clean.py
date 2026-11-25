"""Prepare a user-supplied CSV for solubility prediction.

This public cleaner is deliberately independent of the thesis datasets,
trained weights, and prediction code. It validates molecular identity and
temperature, while preserving every input row and input field. It cannot
calculate applicability-domain similarity because no training reference set
is bundled or selected here.

The CLI recognizes supported downloaded schemas or accepts explicit column
mappings. The standardized input columns are ``smiles``, ``temperature``,
and ``temperature_unit``.
Optional measured target columns must be supplied together:
``solubility``, ``solubility_unit``. Supported target units are mol/L,
mmol/L, g/L, mg/L, ug/L (or μg/L), and log10_mol/L. Mole fraction is not
converted because doing so requires additional composition/density information.

Run from the repository root with its Python environment active:
``python -m data_preparation.clean INPUT.csv OUTPUT.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import Descriptors

from .sources import SOURCES, normalize

REQUIRED = ("smiles", "temperature", "temperature_unit")
OBSERVED = ("solubility", "solubility_unit")
ADDED = (
    "canonical_smiles",
    "temperature_K",
    "input_status",
    "input_issue",
    "duplicate_structure_temperature_input",
    "observed_logS",
    "observation_issue",
)


def finite_number(value: object) -> float:
    """Parse a finite numeric value or raise ``ValueError``."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _structure(smiles: object) -> tuple[Chem.Mol | None, str, str]:
    text = "" if smiles is None else str(smiles).strip()
    molecule = Chem.MolFromSmiles(text) if text else None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None, "", "missing or invalid SMILES"
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    valid = (
        len(Chem.GetMolFrags(molecule)) == 1
        and all(atom.GetFormalCharge() == 0 for atom in molecule.GetAtoms())
        and any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    )
    if not valid:
        return molecule, canonical, "expected a neutral, single-component organic molecule"
    return molecule, canonical, ""


def _temperature(value: object, unit_value: object) -> tuple[float | None, str]:
    try:
        temperature = finite_number(value)
        unit = "" if unit_value is None else str(unit_value).strip().upper()
        if unit in ("C", "°C"):
            temperature += 273.15
        elif unit != "K":
            raise ValueError("temperature_unit must be C or K")
        if temperature <= 0:
            raise ValueError("temperature must be above 0 K")
        return temperature, ""
    except (TypeError, ValueError) as error:
        return None, str(error)


def _measurement(row: Mapping[str, object], molecule: Chem.Mol | None) -> tuple[str, str]:
    raw_value = "" if row.get("solubility") is None else str(row.get("solubility")).strip()
    raw_unit = "" if row.get("solubility_unit") is None else str(row.get("solubility_unit")).strip()
    if not raw_value and not raw_unit:
        return "", ""
    if not raw_value or not raw_unit:
        return "", "provide both solubility and solubility_unit, or leave both blank"
    try:
        value = finite_number(raw_value)
        unit = raw_unit.lower().replace(" ", "")
        if unit in {"log10_mol/l", "log10(mol/l)", "logs(mol/l)"}:
            return str(value), ""
        if unit in {"x", "molefraction", "mole_fraction", "molfraction"}:
            raise ValueError("mole fraction is not converted; required composition or density information is not supplied")
        if value <= 0:
            raise ValueError("solubility concentration must be positive")
        log_value = math.log10(value)
        if unit in {"mol/l", "mol/litre", "mol/liter"}:
            return str(log_value), ""
        if unit in {"mmol/l", "mmol/litre", "mmol/liter"}:
            return str(log_value - 3), ""
        if unit in {"g/l", "g/litre", "g/liter"}:
            mass_offset = 0
        elif unit in {"mg/l", "mg/litre", "mg/liter"}:
            mass_offset = 3
        elif unit in {"ug/l", "μg/l", "µg/l", "mcg/l"}:
            mass_offset = 6
        else:
            raise ValueError("unsupported unit (use mol/L, mmol/L, g/L, mg/L, ug/L, or log10_mol/L)")
        if molecule is None:
            raise ValueError("mass concentration conversion needs a valid molecular structure")
        reported_mass = row.get("solubility_molar_mass_g_mol")
        molar_mass = finite_number(reported_mass) if reported_mass else Descriptors.MolWt(molecule)
        if molar_mass <= 0:
            raise ValueError("molar mass must be positive")
        return str(log_value - mass_offset - math.log10(molar_mass)), ""
    except (TypeError, ValueError, OverflowError) as error:
        return "", str(error)


def prepare(rows: Iterable[Mapping[str, object]], *, water_check=None) -> list[dict[str, object]]:
    """Add standardized fields without dropping or changing any input row."""
    seen: set[tuple[str, float]] = set()
    output: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        molecule, canonical, structure_issue = _structure(row.get("smiles"))
        raw_temperature = row.get("temperature")
        temperature_missing = raw_temperature is None or not str(raw_temperature).strip()
        temperature_k, temperature_issue = (
            (None, "") if temperature_missing else _temperature(raw_temperature, row.get("temperature_unit"))
        )
        solvent_issue = water_check(row) if water_check is not None else ""
        issues = [issue for issue in (structure_issue, temperature_issue, solvent_issue) if issue]
        status = "review" if issues else ("structure_only" if temperature_missing else "ready")
        if temperature_missing:
            issues.append("temperature is missing")
        duplicate = ""
        if not issues and temperature_k is not None:
            key = (canonical, temperature_k)
            duplicate = str(key in seen).lower()
            seen.add(key)
        measurement_molecule = molecule if not structure_issue else None
        log_s, measurement_issue = _measurement(row, measurement_molecule)
        row.update(
            {
                "canonical_smiles": canonical,
                "temperature_K": "" if temperature_k is None else format(temperature_k, ".12g"),
                "input_status": status,
                "input_issue": "; ".join(issues),
                "duplicate_structure_temperature_input": duplicate,
                "observed_logS": log_s,
                "observation_issue": measurement_issue,
            }
        )
        output.append(row)
    return output


def read_rows(path: str | Path, *, cleaned: bool = False, raw: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate a CSV schema while retaining source column order."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        header_set = set(headers)
        if len(headers) != len(header_set):
            raise ValueError("CSV headers must be unique")
        required = () if raw else ((*REQUIRED, *ADDED) if cleaned else REQUIRED)
        missing = [column for column in required if column not in header_set]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))
        collisions = [column for column in ADDED if column in header_set]
        if not cleaned and collisions:
            raise ValueError("Input already contains output columns: " + ", ".join(collisions))
        if not raw and header_set.intersection(OBSERVED) and not set(OBSERVED).issubset(header_set):
            raise ValueError("Supply both solubility and solubility_unit columns when using measurements")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError("Every CSV row must have the same number of fields as the header")
        if not rows:
            raise ValueError("CSV has no records")
    return headers, rows


def write_rows(path: str | Path, headers: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write the supplied fields to a new CSV."""
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def clean_file(input_path: str | Path, output_path: str | Path, **source_options) -> dict[str, int | str]:
    """Validate, clean and export a CSV; the output must not already exist."""
    headers, rows = read_rows(input_path, raw=True)
    source, headers, rows, water_check = normalize(headers, rows, **source_options)
    fields = set(headers)
    if fields.intersection(OBSERVED) and not set(OBSERVED).issubset(fields):
        raise ValueError("Supply both solubility and solubility_unit columns when using measurements")
    cleaned = prepare(rows, water_check=water_check)
    write_rows(output_path, [*headers, *ADDED], cleaned)
    summary = {
        "source": source,
        "input_rows": len(rows),
        "output_rows": len(cleaned),
        "ready_rows": sum(row["input_status"] == "ready" for row in cleaned),
        "structure_only_rows": sum(row["input_status"] == "structure_only" for row in cleaned),
        "review_rows": sum(row["input_status"] == "review" for row in cleaned),
        "duplicate_rows": sum(row["duplicate_structure_temperature_input"] == "true" for row in cleaned),
        "measurement_issues": sum(bool(row["observation_issue"]) for row in cleaned),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input CSV with SMILES and temperature")
    parser.add_argument("output", type=Path, help="new output CSV path")
    parser.add_argument("--source", choices=SOURCES, default="auto",
                        help="CSV layout to use (default detects supported layouts).")
    parser.add_argument("--use-release-molarity", action="store_true",
                        help="Use the derived BigSolDB mol/L column as the comparison label.")
    parser.add_argument("--smiles-column", help="Original column containing molecular SMILES.")
    parser.add_argument("--temperature-column", help="Original column containing temperatures.")
    temperature_units = parser.add_mutually_exclusive_group()
    temperature_units.add_argument("--temperature-unit", choices=("C", "K"))
    temperature_units.add_argument("--temperature-unit-column")
    parser.add_argument("--solubility-column", help="Optional column containing measured solubility.")
    solubility_units = parser.add_mutually_exclusive_group()
    solubility_units.add_argument("--solubility-unit")
    solubility_units.add_argument("--solubility-unit-column")
    parser.add_argument("--solvent-column", help="Optional column for checking whether each solvent is water.")
    args = parser.parse_args()
    try:
        options = vars(args).copy()
        options.pop("input")
        options.pop("output")
        summary = clean_file(args.input, args.output, **options)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "Detected format {source}. Wrote {output_rows} of {input_rows} rows; {ready_rows} ready, "
        "{structure_only_rows} structure-only, {review_rows} need review, {duplicate_rows} "
        "duplicate structure-temperature rows, {measurement_issues} measurement issues.".format(**summary)
    )
    print("Applicability similarity is not assessed by this public cleaner.")


if __name__ == "__main__":
    main()
