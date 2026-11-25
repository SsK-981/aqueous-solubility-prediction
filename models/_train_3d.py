"""Fit the public SchNet variants using the prediction-time geometry protocol."""
from collections import Counter
from copy import deepcopy
from functools import cache
import hashlib
import json
import math

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from .forest import baseline_features
from .sch.conformers import SEEDS as GEOMETRY_SEEDS, generate_conformer
from .sch.model import DESCRIPTORS, CoordinateRegressor, JointReadout, SchNetWithDescriptors


BATCH_SIZE = 4
LEARNING_RATE = 1e-3


@cache
def _geometry(smiles):
    runs = [generate_conformer(smiles, seed) for seed in GEOMETRY_SEEDS]
    for run in runs:
        if run['status'] != 'evaluated':
            return None, run['status']
    if len({run['pairs'] for run in runs}) != 1:
        return None, 'pair_count_disagreement'
    selected = min(runs, key=lambda run: run['energy'])
    atoms = torch.tensor(selected['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(selected['coordinates'], dtype=torch.float32)
    if not atoms.numel() or not ((atoms >= 1) & (atoms < 100)).all():
        return None, 'unsupported_atomic_number'
    if not torch.isfinite(positions).all():
        return None, 'nonfinite_coordinates'
    return (atoms, positions, selected['pairs']), None


@cache
def _descriptors(smiles):
    features = baseline_features(Chem.MolFromSmiles(smiles))
    return [features[name] for name in DESCRIPTORS]


def _prepare(rows, role, custom):
    geometries, context, descriptors, targets = [], [], [], []
    exclusions = Counter()
    for row in rows:
        smiles = row['canonical_smiles']
        geometry, issue = _geometry(smiles)
        if issue is not None:
            exclusions[issue] += 1
            continue
        atoms, positions, pairs = geometry
        temperature = float(row['temperature_K'])
        target = float(row['observed_logS'])
        values = _descriptors(smiles) if custom else []
        if not all(math.isfinite(value) for value in [temperature, target, *values]) or temperature <= 0:
            raise ValueError(f'Invalid numerical input in {role} for {smiles}.')
        geometries.append((atoms, positions))
        context.append([temperature, 1000 / temperature, pairs, 0.0])
        descriptors.append(values)
        targets.append(target)
    if not geometries:
        raise ValueError(f'No {role} rows have usable SchNet geometry. Exclusions: {dict(exclusions)}.')
    counts = {'used': len(geometries), 'excluded': dict(sorted(exclusions.items()))}
    return geometries, np.asarray(context), np.asarray(descriptors), targets, counts


def _tensorize(prepared, context_scaler, descriptor_scaler):
    geometries, context, descriptors, targets, _ = prepared
    context = context_scaler.transform(context)
    context[:, 3] = 0.0
    features = ()
    if descriptor_scaler is not None:
        features = (torch.tensor(descriptor_scaler.transform(descriptors), dtype=torch.float32),)
    return (geometries, torch.tensor(targets, dtype=torch.float32),
            torch.tensor(context, dtype=torch.float32), *features)


def _batches(data, order):
    geometries, targets, *features = data
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        batch = [geometries[index] for index in indices]
        atoms = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
        positions = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True)
        inputs = (atoms, positions, atoms != 0, *(feature[indices] for feature in features))
        yield inputs, targets[indices]


def fit(family, train_rows, validation_rows, checkpoint_root, *, epochs, seed):
    """Fit one fresh seed and export a state dictionary plus training metadata."""
    if family not in ('sch', 'schnet') or epochs < 1:
        raise ValueError('SchNet training requires a sch/schnet family and positive epochs.')
    folder = checkpoint_root / family
    checkpoint = folder / f'seed_{seed}.pt'
    metadata_path = folder / f'seed_{seed}.json'
    if checkpoint.exists() or metadata_path.exists():
        raise ValueError(f'Checkpoint files already exist for {family} seed {seed}.')

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    custom = family == 'schnet'
    train = _prepare(train_rows, 'training', custom)
    validation = _prepare(validation_rows, 'validation', custom)
    context_scaler = StandardScaler().fit(train[1])
    descriptor_scaler = StandardScaler().fit(train[2]) if custom else None
    train_data = _tensorize(train, context_scaler, descriptor_scaler)
    validation_data = _tensorize(validation, context_scaler, descriptor_scaler)
    model = JointReadout(SchNetWithDescriptors(len(DESCRIPTORS))) if custom else CoordinateRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_mae = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, target in _batches(train_data, torch.randperm(len(train[0]), generator=shuffle)):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(*inputs), target)
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite training loss for {family} at epoch {epoch}.')
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            total_error = sum(float((model(*inputs) - target).abs().sum())
                              for inputs, target in _batches(validation_data, torch.arange(len(validation[0]))))
        mae = total_error / len(validation[0])
        if not math.isfinite(mae):
            raise ValueError(f'Non-finite validation MAE for {family} at epoch {epoch}.')
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    metadata = {
        'family': family, 'seed': seed, 'epochs': epochs, 'best_epoch': best_epoch,
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'validation_mae': best_mae, 'train': train[-1], 'validation': validation[-1],
        'geometry_seeds': list(GEOMETRY_SEEDS),
        'global_features': ['temperature_K', '1000 / temperature_K', 'donor_acceptor_pairs', 'masked'],
        'global_mean': context_scaler.mean_.tolist(), 'global_scale': context_scaler.scale_.tolist(),
    }
    if descriptor_scaler is not None:
        metadata.update(descriptors=list(DESCRIPTORS), descriptor_mean=descriptor_scaler.mean_.tolist(),
                        descriptor_scale=descriptor_scaler.scale_.tolist())

    folder.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('xb') as file:
        torch.save(best_state, file)
    metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with metadata_path.open('x', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write('\n')
    return metadata
