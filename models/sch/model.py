"""SchNet encoder and thesis readouts adapted from Panapitiya et al.

SchNet building blocks follow SchNetPack 0.3 (Schuett, Gastegger, Kessel,
Nicoli et al.; MIT), source inspected from its published PyPI wheel:
https://pypi.org/project/schnetpack/0.3/ . This is an independent PyTorch
implementation, not a claim of numerical equivalence to author checkpoints.

Verified old defaults: 25 fixed Gaussian centers on [0,5] Angstrom, widths
equal to center spacing, HARD cutoff <=5 Angstrom, no filter normalization,
five independent interaction blocks, 64 atom features, 128 filters, atomic
embedding indices 0..99, and a 64->32->16->8->1 atomwise MLP averaged over
real atoms. Cosine cutoff is an explicit optional adaptation, NOT the 0.3
default. Pair computation here is dense O(A^2) for nonperiodic molecules;
SchNetPack used neighbor lists. Coordinates are supplied by the caller.
Original coordinates used OpenBabel MMFF94, followed by removing hydrogen;
RDKit ETKDG coordinates are a separate, documented preprocessing adaptation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._temperature import _temperature


class ShiftedSoftplus(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - math.log(2.0)


class SchNetInteraction(nn.Module):
    """SchNetPack 0.3 filter->CFConv->dense residual update, no normalization."""

    def __init__(self, n_gaussians: int = 25):
        super().__init__()
        self.filter_network = nn.Sequential(nn.Linear(n_gaussians, 128), ShiftedSoftplus(), nn.Linear(128, 128))
        self.in2f = nn.Linear(64, 128, bias=False)
        self.f2out = nn.Linear(128, 64)
        self.activation = ShiftedSoftplus()
        self.dense = nn.Linear(64, 64)

    def forward(self, x: Tensor, radial: Tensor, envelope: Tensor, pair_mask: Tensor) -> Tensor:
        filters = self.filter_network(radial) * envelope[..., None]
        # Axis i is the receiving atom, j the sending atom.
        messages = self.in2f(x)[:, None, :, :] * filters
        messages = messages.masked_fill(~pair_mask[..., None], 0.0)
        return self.dense(self.activation(self.f2out(messages.sum(dim=2))))


class SchNetModel(nn.Module):
    """Nonperiodic SchNet matching the author's selected 0.3 architecture.

    Inputs: atomic_numbers [B,A], positions [B,A,3] in Angstrom, mask [B,A].
    Padding atomic numbers/positions are ignored. Real atomic numbers must be
    1..99 (old max_z=100 embedding). Every molecule needs >=1 real atom.
    ``cutoff_function='cosine'`` changes the old default and must be disclosed.
    """

    def __init__(self, temperature: bool = False, *, cutoff_function: str = "hard"):
        super().__init__()
        if cutoff_function not in ("hard", "cosine"):
            raise ValueError("cutoff_function must be hard or cosine")
        self.temperature_enabled = temperature
        self.cutoff_function = cutoff_function
        self.cutoff = 5.0
        self.embedding = nn.Embedding(100, 64, padding_idx=0)
        self.register_buffer("gaussian_offsets", torch.linspace(0.0, 5.0, 25))
        self.register_buffer("gaussian_width", torch.full((25,), 5.0 / 24.0))
        self.interactions = nn.ModuleList([SchNetInteraction() for _ in range(5)])
        self.atomwise = nn.Sequential(
            nn.Linear(64 + (2 if temperature else 0), 32), ShiftedSoftplus(),
            nn.Linear(32, 16), ShiftedSoftplus(), nn.Linear(16, 8),
            ShiftedSoftplus(), nn.Linear(8, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, atomic_numbers: Tensor, positions: Tensor, mask: Tensor,
                temperature: Tensor | None = None) -> Tensor:
        if (atomic_numbers.ndim != 2 or positions.shape != (*atomic_numbers.shape, 3)
                or mask.shape != atomic_numbers.shape or not atomic_numbers.numel()):
            raise ValueError("Expected atomic_numbers[B,A], positions[B,A,3], mask[B,A]")
        if atomic_numbers.dtype not in (torch.int32, torch.int64):
            raise ValueError("Atomic numbers must be integer tensors")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask must contain only 0/1 values")
        real = mask.bool()
        if not real.any(dim=1).all():
            raise ValueError("Each molecule must contain at least one real atom")
        if ((atomic_numbers[real] < 1) | (atomic_numbers[real] >= 100)).any():
            raise ValueError("Real atomic numbers must be in 1..99 (original max_z=100)")
        if not torch.isfinite(positions[real]).all():
            raise ValueError("Real atom coordinates must be finite")
        z = torch.where(real, atomic_numbers, 0)
        xyz = torch.where(real[..., None], positions, 0.0)
        xyz = xyz.to(dtype=self.embedding.weight.dtype)
        n_atoms = z.shape[1]
        pair_mask = real[:, :, None] & real[:, None, :]
        pair_mask = pair_mask & ~torch.eye(n_atoms, device=z.device, dtype=torch.bool)[None]
        distances = torch.linalg.vector_norm(xyz[:, :, None, :] - xyz[:, None, :, :], dim=-1)
        radial = torch.exp(-0.5 * ((distances[..., None] - self.gaussian_offsets) / self.gaussian_width).square())
        if self.cutoff_function == "hard":
            envelope = (distances <= self.cutoff).to(xyz.dtype)
        else:
            envelope = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
            envelope = envelope * (distances < self.cutoff)
        x = self.embedding(z)
        for interaction in self.interactions:
            x = (x + interaction(x, radial, envelope, pair_mask)).masked_fill(~real[..., None], 0.0)
        x = _temperature(x, temperature, self.temperature_enabled)
        atom_values = self.atomwise(x).squeeze(-1).masked_fill(~real, 0.0)
        return atom_values.sum(dim=1) / real.sum(dim=1).to(atom_values.dtype)


# Fixed order from the thesis validation-selected descriptor panel.
DESCRIPTORS = ("FractionCSP3", "HBD", "MolLogP", "MolWt", "TPSA")

class CoordinateRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = SchNetModel(temperature=True)
        self.pair_effect = nn.Linear(1, 1, bias=False)

    def forward(self, z, pos, mask, context):
        return self.network(z, pos, mask, context[:, :2]) + self.pair_effect(context[:, 2:3]).flatten()


class SchNetWithDescriptors(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.base = CoordinateRegressor()
        self.extra = nn.Linear(n_features, 1, bias=False)
        nn.init.zeros_(self.extra.weight)

    def forward(self, z, positions, mask, context, features):
        return self.base(z, positions, mask, context) + self.extra(features).flatten()


class JointReadout(nn.Module):
    """Add descriptors to the first nonlinear atomwise layer before its activation."""

    def __init__(self, additive):
        super().__init__()
        self.base = additive.base
        self.descriptor_projection = nn.Linear(len(DESCRIPTORS), 32, bias=False)
        nn.init.zeros_(self.descriptor_projection.weight)

    def forward(self, z, positions, mask, context, descriptors):
        descriptor_terms = self.descriptor_projection(descriptors)[:, None, :]

        # Limit the hook to this call. The original geometric encoder and
        # temperature inputs are shared with the additive control.
        def condition_readout(_layer, _inputs, output):
            return output + descriptor_terms

        handle = self.base.network.atomwise[0].register_forward_hook(condition_readout)
        try:
            return self.base(z, positions, mask, context)
        finally:
            handle.remove()
