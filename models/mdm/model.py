"""Molecular descriptor model adapted from Panapitiya et al.

See THIRD_PARTY_NOTICES.md for the Battelle notice.
"""

import torch
from torch import nn

MDM_DROPOUTS = (0.10696194799818459, 0.6033824611348487)


class DescriptorModel(nn.Module):
    """Author MDM dense topology port; callers supply (B,n_features) inputs.

    Eight-descriptor RDKit inputs are an adaptation, not the author's full MDM
    feature pipeline. Match preprocessing to the fitted model explicitly.
    """

    def __init__(self, n_features):
        super().__init__()
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        self.n_features = n_features
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout1 = nn.Dropout(MDM_DROPOUTS[0])
        self.fc2 = nn.Linear(128, 576)
        self.dropout2 = nn.Dropout(MDM_DROPOUTS[1])
        self.out = nn.Linear(576, 1)
        # Match Keras Dense defaults (Glorot uniform kernel and zero bias).
        # Random number streams differ between frameworks; draws are not equal.
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(f"Expected descriptor shape (B,{self.n_features})")
        x = self.dropout1(torch.sigmoid(self.fc1(features)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        return self.out(x).squeeze(-1)
