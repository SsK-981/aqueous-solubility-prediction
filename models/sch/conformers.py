"""Geometry preparation for the SOMAS neural models."""
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolTransforms

BUDGET = 96


SEEDS = (42, 101)


CUTOFF = (2.5, 3.5, 120.0)


MAX_HEAVY_ATOMS = 45


def _factory():
    return ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )


def _sites(molecule, factory):
    features = factory.GetFeaturesForMol(molecule)
    donors = {f.GetAtomIds()[0] for f in features
              if f.GetFamily() == "Donor" and len(f.GetAtomIds()) == 1}
    acceptors = {f.GetAtomIds()[0] for f in features
                 if f.GetFamily() == "Acceptor" and len(f.GetAtomIds()) == 1}
    rule = Chem.MolFromSmarts("[N;X3;H0;+0](-[CX4])(-[CX4])-[CX4]")
    acceptors.update(match[0] for match in molecule.GetSubstructMatches(rule))
    return donors, acceptors


def generate_conformer(smiles, seed):
    factory = _factory()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"seed": seed, "status": "invalid_structure"}
    if (len(Chem.GetMolFrags(molecule)) != 1
            or any(atom.GetFormalCharge() for atom in molecule.GetAtoms())
            or molecule.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS):
        return {"seed": seed, "status": "out_of_scope"}
    donors, acceptors = _sites(molecule, factory)
    explicit = Chem.AddHs(molecule)
    triples = []
    for donor in donors:
        hydrogens = [neighbor.GetIdx() for neighbor in explicit.GetAtomWithIdx(donor).GetNeighbors()
                     if neighbor.GetAtomicNum() == 1]
        for acceptor in acceptors:
            if donor != acceptor and len(Chem.GetShortestPath(molecule, donor, acceptor)) >= 4:
                triples.extend((donor, hydrogen, acceptor) for hydrogen in hydrogens)
    pairs = sorted({(donor, acceptor) for donor, _, acceptor in triples})
    if not triples:
        return {"seed": seed, "status": "no_eligible_pair", "pairs": 0, "fraction": 0.0}
    if not AllChem.MMFFHasAllMoleculeParams(explicit):
        return {"seed": seed, "status": "missing_mmff_parameters", "pairs": len(pairs)}
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1
    conformer_ids = list(AllChem.EmbedMultipleConfs(explicit, numConfs=BUDGET, params=params))
    if not conformer_ids:
        return {"seed": seed, "status": "embedding_failed", "pairs": len(pairs)}
    optimized = AllChem.MMFFOptimizeMoleculeConfs(
        explicit, numThreads=1, maxIters=500, mmffVariant="MMFF94s"
    )
    converged = [(cid, energy) for cid, (status, energy) in zip(conformer_ids, optimized)
                 if status == 0]
    if not converged:
        return {"seed": seed, "status": "no_converged_conformer",
                "pairs": len(pairs), "embedded": len(conformer_ids)}
    minimum = min(energy for _, energy in converged)
    retained = [(cid, energy) for cid, energy in converged if energy - minimum <= 5.0]
    hits = set()
    for cid, _ in retained:
        conf = explicit.GetConformer(cid)
        for donor, hydrogen, acceptor in triples:
            ha = (conf.GetAtomPosition(hydrogen) - conf.GetAtomPosition(acceptor)).Length()
            da = (conf.GetAtomPosition(donor) - conf.GetAtomPosition(acceptor)).Length()
            angle = rdMolTransforms.GetAngleDeg(conf, donor, hydrogen, acceptor)
            if ha <= CUTOFF[0] and da <= CUTOFF[1] and angle >= CUTOFF[2]:
                hits.add(cid)
                break
    best_cid, best_energy = min(retained, key=lambda item: item[1])
    conf = explicit.GetConformer(best_cid)
    coordinates = np.asarray(conf.GetPositions(), dtype=np.float32)
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in explicit.GetAtoms()], dtype=np.int8)
    return {
        "seed": seed, "status": "evaluated", "pairs": len(pairs),
        "fraction": len(hits) / len(retained), "embedded": len(conformer_ids),
        "converged": len(converged), "retained": len(retained),
        "atomic_numbers": atomic_numbers, "coordinates": coordinates,
        "energy": np.float32(best_energy),
    }
