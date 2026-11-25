"""Delaney ESOL with the Pat Walters RDKit refit."""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

AROMATIC_ATOM = Chem.MolFromSmarts("a")


def esol_log_s(smiles: str) -> float:
    """Return the fixed RDKit-refit ESOL estimate in log10(mol/L)."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")
    aromatic_atom_fraction = len(molecule.GetSubstructMatches(AROMATIC_ATOM)) / molecule.GetNumAtoms()
    return (
        0.26121066137801696
        - 0.7416739523408995 * Crippen.MolLogP(molecule)
        - 0.0066138847738667125 * Descriptors.MolWt(molecule)
        + 0.003451545565957996 * Lipinski.NumRotatableBonds(molecule)
        - 0.42624840441316975 * aromatic_atom_fraction
    )
