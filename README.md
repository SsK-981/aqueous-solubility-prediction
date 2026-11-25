# Aqueous solubility prediction

## Quick start

1. From the repository root, set up Python 3.12.

   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   mkdir -p data
   ```

2. Download `SOMAS.csv` from the [Figshare dataset](https://doi.org/10.6084/m9.figshare.35118649), save it in `data/`, and prepare it.

   ```sh
   python -m data_preparation.clean data/SOMAS.csv data/somas_clean.csv --source somas
   ```

   See the [data guide](data_preparation/README.md) for other CSVs in the download.

3. Run a model. ESOL runs immediately. For a fitted model, [check that its files are in `checkpoints/`](models/README.md#checkpoints), then change `--model` and the output filename.

   ```sh
   python -m models.predict --model esol --input data/somas_clean.csv --output data/somas_esol.csv
   ```

The output CSV includes a prediction status for each input row and solubility in log₁₀(mol/L) and mol/L for scored rows.

## Models

Each folder has a prediction command. The fitted model families also have a training command. [Run or train a model](models/README.md).

| Model | `--model` | Inputs |
|---|---|---|
| [Molecular descriptor model](models/mdm/) | `mdm` | Eight RDKit descriptors |
| [Graph neural network](models/gnn/) | `gnn` | Atoms and bonds |
| [SMILES sequence model](models/smi/) | `smi` | SMILES |
| [Plain SchNet](models/sch/) | `sch` | Generated 3D coordinates, temperature, and donor–acceptor pair count |
| [Custom SchNet](models/schnet/) | `schnet` | Plain SchNet inputs plus five RDKit descriptors |
| [Random Forest](models/rf/) | `rf` | RDKit descriptors, Morgan fingerprints, and temperature |
| [ESOL](models/esol/) | `esol` | Structure-derived descriptors |

Layout reference · [Panapitiya et al.’s repository](https://github.com/pnnl/solubility-prediction-paper)

[Third-party attribution](THIRD_PARTY_NOTICES.md)
