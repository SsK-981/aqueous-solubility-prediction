"""SMILES sequence adaptation of the Panapitiya et al. model.

SMI deliberately has no padding mask, matching the author's unmasked Keras
embedding. PAD therefore remains trainable and changing padded length can
change predictions. Freeze ONE maxlen in preprocessing; reject longer inputs
unless truncation is explicitly requested. Train-only vocabulary with UNK is
an adaptation from the author's vocabulary built across all three splits.
PyTorch LSTM uses (input,forget,cell,output) gates, corresponding to Keras'
(i,f,c,o) after transposing kernels and handling the two PyTorch biases.
Old Keras commonly defaults to hard_sigmoid recurrent gates whereas nn.LSTM
uses sigmoid; initializers/backend arithmetic also differ. No numerically
identical Keras checkpoint conversion has been verified. Train these models
as adaptations; do not load legacy pickles/checkpoints into them implicitly.

Temperature conditioning is an extension absent from the author's models.
It expects two already prepared values [Temperature_K,1000/Temperature_K]
standardized per molecule. Scaling must be fitted on inner training data only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._temperature import _temperature

PAD_ID = 0
UNK_ID = 1


def tokenize_smiles(smiles: str) -> list[str]:
    """Character tokenization exactly as in the author's SMI preprocessing."""
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("SMILES must be a nonempty string")
    return list(smiles)


def build_vocabulary(smiles: Iterable[str]) -> dict[str, int]:
    """Fit sorted character IDs >=2; caller MUST provide training SMILES only.

    0=PAD and 1=UNK are reserved numeric IDs, not dictionary keys. Persist this
    dictionary rather than rebuilding it on validation/test structures.
    """
    chars = {char for smi in smiles for char in tokenize_smiles(smi)}
    if not chars:
        raise ValueError("Cannot fit a vocabulary from empty training data")
    return {char: i for i, char in enumerate(sorted(chars), start=2)}


def encode_sequences(
    smiles: Iterable[str], vocabulary: Mapping[str, int], maxlen: int,
    *, truncate: bool = False, return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Encode and right-pad to a frozen maxlen; unseen characters become UNK.

    Longer sequences raise by default, avoiding silent removal of chemistry.
    ``truncate=True`` retains the prefix explicitly. A report counts unknown
    characters/sequences and truncations; it never fits or mutates vocabulary.
    Maxlen may be a fixed predeclared cap or derived from training structures;
    if derived from all target-free structures, disclose that protocol choice.
    """
    if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen <= 0:
        raise ValueError("maxlen must be a positive integer")
    if (any(not isinstance(k, str) or len(k) != 1 for k in vocabulary)
            or sorted(vocabulary.values()) != list(range(2, len(vocabulary) + 2))):
        raise ValueError("Vocabulary must map characters to contiguous IDs starting at 2")
    rows = [tokenize_smiles(smi) for smi in smiles]
    encoded = np.zeros((len(rows), maxlen), dtype=np.int64)
    unknowns: dict[str, int] = {}
    unknown_sequences = truncated = 0
    for i, row in enumerate(rows):
        if len(row) > maxlen:
            if not truncate:
                raise ValueError(f"Sequence {i} has length {len(row)} > frozen maxlen {maxlen}")
            truncated += 1
        has_unknown = False
        for j, char in enumerate(row[:maxlen]):
            encoded[i, j] = vocabulary.get(char, UNK_ID)
            if char not in vocabulary:
                has_unknown = True
                unknowns[char] = unknowns.get(char, 0) + 1
        unknown_sequences += int(has_unknown)
    report = {"n_sequences": len(rows), "maxlen": maxlen,
              "unknown_characters": dict(sorted(unknowns.items())),
              "unknown_character_count": sum(unknowns.values()),
              "unknown_sequence_count": unknown_sequences,
              "truncated_sequence_count": truncated,
              "padding": "right; unmasked and trainable"}
    return (encoded, report) if return_report else encoded


class SequenceModel(nn.Module):
    """Author SMI architecture; output shape [B], optional temperature extension."""

    def __init__(self, vocab_size: int, temperature: bool = False):
        super().__init__()
        if vocab_size < 3:
            raise ValueError("vocab_size must include PAD, UNK and at least one character")
        self.temperature_enabled = temperature
        # No padding_idx: original Embedding(mask_zero=False) learns PAD too.
        self.embedding = nn.Embedding(vocab_size, 960)
        self.lstm = nn.LSTM(960, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.08607118576024131)
        self.bidirectional_lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.4730059911045743)
        self.dense1 = nn.Linear(512 + (2 if temperature else 0), 704)
        self.dense2 = nn.Linear(704, 640)
        self.dropout3 = nn.Dropout(0.186637772607526)
        self.output = nn.Linear(640, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        for recurrent in (self.lstm, self.bidirectional_lstm):
            for name, parameter in recurrent.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(parameter)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)
                    if name.startswith("bias_ih"):
                        with torch.no_grad():
                            parameter[256:512].fill_(1.0)
        for layer in (self.dense1, self.dense2, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: Tensor, temperature: Tensor | None = None) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0 or tokens.shape[0] == 0:
            raise ValueError("tokens must have nonempty shape [B,L]")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must contain integer IDs")
        if (tokens < 0).any() or (tokens >= self.embedding.num_embeddings).any():
            raise ValueError("Token ID outside fitted vocabulary")
        sequence, _ = self.lstm(self.embedding(tokens))
        _, (hidden, _) = self.bidirectional_lstm(self.dropout1(sequence))
        # Final state of each direction, not the final timestep's reverse state.
        representation = self.dropout2(torch.cat((hidden[0], hidden[1]), dim=-1))
        representation = _temperature(representation, temperature, self.temperature_enabled)
        hidden = torch.sigmoid(self.dense1(representation))
        hidden = self.dropout3(F.relu(self.dense2(hidden)))
        return self.output(hidden).squeeze(-1)
