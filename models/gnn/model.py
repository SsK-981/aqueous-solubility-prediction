"""PyTorch ports of the published GNN architecture.

Source: solubility-prediction-paper-master, ACS Omega DOI 10.1021/acsomega.2c00642.
Original source SHA256 (unmodified author files):
gnn/gnn_model.py: e1783dfd130122990932c49d5c1f2c77dc2a6d84083f0a15b2c63e1aa272fdec
gnn/gnn_utils.py: 141d430559382c47ecd96938d856765ef2eb3ebc7ae5979ed5932331a1fcf206

Only active author layers are instantiated. Author bond features are returned but
are not consumed by the author GNN. Temperature adds two caller-prepared columns
after sum pooling. The GraphRegressor class is the separate thesis model used in the
matched hydrogen-bond contact experiment.

Solubility Prediction Models
Copyright © 2021, Battelle Memorial Institute

1. Battelle Memorial Institute (hereinafter Battelle) hereby grants permission to any person or entity lawfully obtaining a copy of this software and associated documentation files (hereinafter “the Software”) to redistribute and use the Software in source and binary forms, with or without modification. Such person or entity may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and may permit others to do so, subject to the following conditions:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Other than as used herein, neither the name Battelle Memorial Institute or Battelle may be used in any form whatsoever without the express written consent of Battelle.

2. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, GCNConv, global_add_pool, global_mean_pool

NODE_FEATURES = 65
EDGE_FEATURES = 6
ATOM_SYMBOLS = (
    'Ag', 'Al', 'As', 'B', 'Br', 'C', 'Ca', 'Cd', 'Cl', 'Cu', 'F',
    'Fe', 'Ge', 'H', 'Hg', 'I', 'K', 'Li', 'Mg', 'Mn', 'N', 'Na',
    'O', 'P', 'Pb', 'Pt', 'S', 'Se', 'Si', 'Sn', 'Sr', 'Tl', 'Zn', 'Unknown',
)
GNN_DROPOUTS = (0.9444250299450242, 0.8341272742321129,
                0.7675340644596443, 0.6040220843354102,
                0.4813038851902818, 0.015105134306121593,
                0.3431295462686682)


def _one_hot(value, choices, unknown=False):
    if value not in choices:
        if not unknown:
            raise ValueError(f"Unsupported atom degree {value}; expected {choices}")
        value = choices[-1]
    return [value == item for item in choices]


def atom_features(atom):
    """Exact author default atom_features: implicit H, no chirality, 65 columns."""
    hybridization = Chem.rdchem.HybridizationType
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS, unknown=True)
        + _one_hot(atom.GetDegree(), list(range(11)))
        + _one_hot(atom.GetValence(Chem.ValenceType.IMPLICIT), list(range(7)), unknown=True)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + _one_hot(atom.GetHybridization(), [hybridization.SP, hybridization.SP2,
                   hybridization.SP3, hybridization.SP3D, hybridization.SP3D2],
                   unknown=True)
        + [atom.GetIsAromatic()]
        + _one_hot(atom.GetTotalNumHs(), list(range(5)), unknown=True)
    )


def bond_features(bond):
    """Exact author default six bond columns, with no chirality."""
    kind = bond.GetBondType()
    bond_type = Chem.rdchem.BondType
    return [kind == bond_type.SINGLE, kind == bond_type.DOUBLE,
            kind == bond_type.TRIPLE, kind == bond_type.AROMATIC,
            bond.GetIsConjugated(), bond.IsInRing()]


def encode_graph(smiles):
    """Build author features and directed bonds; invalid/empty input raises.

    Isolated atoms are supported, including a molecule with no edges. No atom,
    fragment or molecule is dropped, canonicalized, neutralized, or filtered.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty molecular SMILES: {smiles!r}")
    nodes = [atom_features(atom) for atom in mol.GetAtoms()]
    pairs, edges = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend(((start, end), (end, start)))
        features = bond_features(bond)
        edges.extend((features, features))
    return Data(x=torch.tensor(nodes, dtype=torch.float32),
                edge_index=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous(),
                edge_attr=torch.tensor(edges, dtype=torch.float32).reshape(-1, EDGE_FEATURES))


class GraphModel(nn.Module):
    """Active author GNN; input is a PyG Data/Batch, output shape (B,).

    For temperature=True provide data.temperature of shape (B, 2). For an
    individual Data intended for Batch.from_data_list, store shape (1, 2).
    """

    def __init__(self, temperature=False):
        super().__init__()
        self.temperature = bool(temperature)
        self.n_features = NODE_FEATURES
        self.gcn1 = GCNConv(NODE_FEATURES, 256, cached=False)
        self.gcn2 = GCNConv(256, 320, cached=False)
        self.gcn21 = GCNConv(320, 448, cached=False)
        self.gcn32 = GCNConv(448, 256, cached=False)
        self.gdo1, self.gdo2, self.gdo21, self.gdo32, self.edo1, self.dropout1, self.dropout2 = (
            nn.Dropout(p) for p in GNN_DROPOUTS)
        self.gact1, self.gact2, self.gact21 = nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()
        self.gact32 = nn.SELU()
        self.ecn1 = EdgeConv(nn.Sequential(nn.Linear(NODE_FEATURES * 2, 256),
                                         nn.ReLU(), nn.Linear(256, 256)))
        self.eact1 = nn.ReLU()
        self.fc1 = nn.Linear(512 + 2 * self.temperature, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.Sigmoid()
        self.out2 = nn.Linear(256, 1)

    def forward(self, data):
        node_x, edge_index = data.x, data.edge_index
        if node_x.ndim != 2 or node_x.shape[1] != NODE_FEATURES or node_x.shape[0] == 0:
            raise ValueError("Graph nodes must have shape (N,65), N > 0")
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(node_x.shape[0], dtype=torch.long, device=node_x.device)
        x1 = self.gdo1(self.gact1(self.gcn1(node_x, edge_index)))
        x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)))
        x1 = self.gdo21(self.gact21(self.gcn21(x1, edge_index)))
        x1 = self.gdo32(self.gact32(self.gcn32(x1, edge_index)))
        x2 = self.edo1(self.eact1(self.ecn1(node_x, edge_index)))
        x3 = global_add_pool(torch.cat((x1, x2), dim=1), batch)
        if self.temperature:
            temperature = getattr(data, 'temperature', None)
            if temperature is None or temperature.shape != (x3.shape[0], 2):
                raise ValueError("Temperature model requires data.temperature with shape (B,2)")
            x3 = torch.cat((x3, temperature.to(device=x3.device, dtype=x3.dtype)), dim=1)
        x3 = self.dropout1(self.act1(self.fc1(x3)))
        x3 = self.dropout2(self.act2(self.fc2(x3)))
        return self.out2(x3).squeeze(-1)


GNN = GraphModel


# GNN used in the matched hydrogen-bond contact experiment.
class GraphRegressor(nn.Module):
    def __init__(self, node_features):
        super().__init__()
        self.conv1 = GCNConv(node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.head = nn.Sequential(nn.Linear(68, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, graph, context):
        x = self.conv1(graph.x, graph.edge_index).relu()
        x = self.conv2(x, graph.edge_index).relu()
        x = self.conv3(x, graph.edge_index).relu()
        return self.head(torch.cat([global_mean_pool(x, graph.batch), context], dim=1)).flatten()
