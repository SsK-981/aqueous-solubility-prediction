# Run the models

Run commands from the repository root after [setup](../README.md#quick-start).

## Checkpoints

The ZIP with checkpoints includes the fitted models in `checkpoints/`. For a source-only copy, download the ZIP for your model from **Releases → Assets** and extract it into `checkpoints/` at the repository root.

| Release asset | `--model` | Folders under `checkpoints/` |
|---|---|---|
| `neural-checkpoints.zip` | `mdm`, `gnn`, `smi` | `mdm/`, `gnn/`, `smi/` |
| `schnet-checkpoints.zip` | `sch`, `schnet` | `sch/`, `schnet/` |
| `random-forest-checkpoints.zip` | `rf` | `rf/` |

ESOL needs no checkpoint. Load Random Forest checkpoints from a trusted source.

## Predict one molecule

```sh
python -m models.mdm.predict --smiles 'CCO'
python -m models.sch.predict --smiles 'Nc1cncc(Cl)n1' --celsius 25
```

Use neutral, single-component organic molecules. Random Forest and both SchNet variants require temperature. Both SchNet variants also require an eligible donor–acceptor pair and a converged conformer.

## Score a CSV

```sh
python -m models.mdm.predict --input data/somas_clean.csv --output data/somas_mdm.csv
```

Use a [prepared CSV](../data_preparation/README.md) and a new output filename. The command reports MAE for scored rows with compatible measurements.

## Train a model

```sh
python -m models.mdm.train --input data/somas_clean.csv --checkpoints checkpoints/new-fits
python -m models.mdm.predict --checkpoints checkpoints/new-fits --smiles 'CCO'
```

Replace `mdm` with `gnn`, `smi`, `sch`, `schnet`, or `rf`. Use a prepared CSV with measured solubility. For either SchNet variant, both splits need structures with usable conformers. By default, training groups records by compound identity and reserves about 15% of the groups for validation. Pass `--validation` to use a separate prepared CSV with distinct compounds.

These commands fit new weights. The release assets contain the saved benchmark fits. ESOL uses a fixed equation and has a prediction command only.
