"""Encoder architectures for the semantic (text-motion matching) verifier.

Same architecture family as the text-to-motion / MotionGPT evaluator:
  - ``MovementConvEncoder`` / ``MovementConvDecoder``: 1D-conv "movement" codec that
    downsamples a window of raw 36-dim motion frames by 4x into a latent sequence
    (trained as an autoencoder in the "decomp" stage).
  - ``TextEncoderBiGRUCo`` / ``MotionEncoderBiGRUCo``: bidirectional-GRU encoders that
    map a caption / a movement-latent sequence to a shared 512-d co-embedding space
    (trained contrastively in the "match" stage, with the movement encoder frozen).

Extracted from the original ``train_evaluator.py`` so training and inference share one
definition of the four encoder classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def init_weight(m: nn.Module) -> None:
    """Xavier-normal init for conv/linear layers, matching the original evaluator."""
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    """Downsamples (B, T, input_size) motion by 4x into (B, T/4, output_size) latents."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    """Upsamples (B, T/4, input_size) latents back to (B, T, output_size) motion."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    """Bi-GRU over movement-latent frames -> single 512-d motion co-embedding."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs: torch.Tensor, m_lens) -> torch.Tensor:
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        emb = input_embs
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class TextEncoderBiGRUCo(nn.Module):
    """Bi-GRU over (word_embedding + pos_onehot) tokens -> 512-d text co-embedding."""

    def __init__(self, word_size: int, pos_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs: torch.Tensor, pos_onehot: torch.Tensor, cap_lens) -> torch.Tensor:
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist() if torch.is_tensor(cap_lens) else list(cap_lens)
        emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


__all__ = [
    "init_weight",
    "MovementConvEncoder",
    "MovementConvDecoder",
    "MotionEncoderBiGRUCo",
    "TextEncoderBiGRUCo",
]
