# Prepare a dataset

[Download a CSV](../README.md#quick-start), save it in `data/`, and run from the repository root.

| CSV in `data/` | `--source` |
|---|---|
| `SOMAS.csv` | `somas` |
| `BigSolDB_v2.2.csv` | `bigsoldb` |
| `AqSolDBc.csv` | `aqsoldbc` |
| `DLS100_56_compounds.csv` | `dls100` |

```sh
python -m data_preparation.clean data/SOMAS.csv data/somas_clean.csv --source somas
```

Change the input, output, and source for your dataset. Use the source filenames listed above to prepare input for `models.predict`. For BigSolDB, add `--use-release-molarity` to score against the release’s derived molar measurements. Prediction skips non-water rows.

## Use your own CSV

Use `smiles` and, for temperature-aware models, `temperature` with `temperature_unit` (`C` or `K`). For scoring against measurements, add both `solubility` and `solubility_unit` (`mol/L`, `mmol/L`, `g/L`, `mg/L`, `ug/L`, or `log10_mol/L`).

```sh
python -m data_preparation.clean data/input.csv data/input_clean.csv
```

Run `python -m data_preparation.clean --help` to map different column names. Use a new output filename, then [run a model on the prepared CSV](../models/README.md#score-a-csv).
