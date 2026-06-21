"""Minimal inference API for the semantic (text-motion matching) verifier.

Loads the trained ``finest.tar`` (text_encoder + motion_encoder + movement_encoder)
plus the mean/std normalization stats and GloVe vocabulary, then exposes a single
clean entry point:

    evaluator = load_evaluator(ckpt, meta_dir, glove_dir, device="cuda")
    distance = evaluator.score(motion_36d, caption)   # lower = better match

``score`` returns the L2 distance between the 512-d text and motion co-embeddings
(the same "matching score" used by MotionGPT's FID/R-Precision/Matching-Score
evaluation, here exposed for best-of-N candidate selection).

Self-contained: torch + numpy only, no spaCy/regex POS tagger. Captions are
lower-cased and split on whitespace; words missing from the GloVe vocabulary (or
without an explicit ``word/POS`` suffix) fall back to the ``OTHER`` POS tag and the
``unk`` embedding, exactly like ``WordVectorizer`` does during training.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np
import torch

try:  # works both as a package import (verifiers.semantic.inference) and as a script
    from .models import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
    from .train_evaluator import WordVectorizer, root_pos_to_vel_np
except ImportError:
    from models import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
    from train_evaluator import WordVectorizer, root_pos_to_vel_np

_WORD_RE = re.compile(r"[a-zA-Z]+")

# Architecture dims must match configs/evaluator.yaml (arch section).
_DEFAULT_DIMS = dict(
    motion_input_dim=36,
    dim_movement_enc_hidden=512,
    dim_movement_latent=512,
    dim_word=300,
    dim_pos_ohot=15,
    dim_text_hidden=512,
    dim_motion_hidden=1024,
    dim_coemb_hidden=512,
)


def _simple_tokenize(caption: str) -> list[str]:
    """Lower-case word tokens (no POS tag) -> WordVectorizer falls back to OTHER."""
    return _WORD_RE.findall(caption.lower())


class SemanticEvaluator:
    """Text-motion matching scorer built from a trained finest.tar checkpoint."""

    def __init__(
        self,
        text_encoder: TextEncoderBiGRUCo,
        motion_encoder: MotionEncoderBiGRUCo,
        movement_encoder: MovementConvEncoder,
        word_vectorizer: WordVectorizer,
        mean: np.ndarray,
        std: np.ndarray,
        device: Union[str, torch.device] = "cpu",
        unit_length: int = 4,
        root_pos_dims: int = 3,
        use_root_vel: bool = True,
        max_text_len: int = 50,
    ):
        self.device = torch.device(device)
        self.text_encoder = text_encoder.to(self.device).eval()
        self.motion_encoder = motion_encoder.to(self.device).eval()
        self.movement_encoder = movement_encoder.to(self.device).eval()
        self.word_vectorizer = word_vectorizer
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.unit_length = unit_length
        self.root_pos_dims = root_pos_dims
        self.use_root_vel = use_root_vel
        self.max_text_len = max_text_len

    @torch.no_grad()
    def embed_motion(self, motion_36d: np.ndarray) -> torch.Tensor:
        """Encode a single (T, 36) motion into a (512,) embedding."""
        motion = np.asarray(motion_36d, dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected motion of shape (T, {self.mean.shape[0]}), got {motion.shape}")

        if self.use_root_vel:
            motion = root_pos_to_vel_np(motion, self.root_pos_dims)

        m_length = (len(motion) // self.unit_length) * self.unit_length
        m_length = max(m_length, self.unit_length)
        motion = motion[:m_length]

        motion = (motion - self.mean) / self.std
        motion_t = torch.from_numpy(motion).float().unsqueeze(0).to(self.device)  # (1, T, 36)

        movement = self.movement_encoder(motion_t)
        m_len_enc = torch.tensor([m_length // self.unit_length], device=self.device)
        motion_emb = self.motion_encoder(movement, m_len_enc)
        return motion_emb.squeeze(0)

    @torch.no_grad()
    def embed_text(self, caption: str) -> torch.Tensor:
        """Encode a caption into a (512,) embedding."""
        tokens = _simple_tokenize(caption) or ["unk"]

        if len(tokens) < self.max_text_len:
            tokens = ["sos"] + tokens + ["eos"]
            sent_len = len(tokens)
            tokens = tokens + ["unk"] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[: self.max_text_len]
            tokens = ["sos"] + tokens + ["eos"]
            sent_len = len(tokens)

        word_embeddings = []
        pos_one_hots = []
        for token in tokens:
            word_emb, pos_oh = self.word_vectorizer[token]
            word_embeddings.append(word_emb[None, :])
            pos_one_hots.append(pos_oh[None, :])
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)

        word_emb_t = torch.from_numpy(word_embeddings).float().unsqueeze(0).to(self.device)
        pos_oh_t = torch.from_numpy(pos_one_hots).float().unsqueeze(0).to(self.device)
        cap_lens = torch.tensor([sent_len])

        text_emb = self.text_encoder(word_emb_t, pos_oh_t, cap_lens)
        return text_emb.squeeze(0)

    @torch.no_grad()
    def score(self, motion_36d: np.ndarray, caption: str) -> float:
        """L2 distance between the motion and text co-embeddings (lower = better match)."""
        motion_emb = self.embed_motion(motion_36d)
        text_emb = self.embed_text(caption)
        return torch.norm(motion_emb - text_emb, p=2).item()


def load_evaluator(
    checkpoint: Union[str, Path],
    meta_dir: Union[str, Path],
    glove_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    **dims,
) -> SemanticEvaluator:
    """Load the trained semantic evaluator for inference.

    Args:
        checkpoint: path to ``.../text_mot_match/model/finest.tar``
            (keys: ``text_encoder``, ``motion_encoder``, ``movement_encoder``).
        meta_dir: directory containing ``mean.npy`` and ``std.npy``
            (``.../Comp_v6_KLD01/meta``).
        glove_dir: directory containing ``our_vab_data.npy`` / ``our_vab_words.pkl`` /
            ``our_vab_idx.pkl``.
        device: torch device for inference.
        **dims: optional architecture-dim overrides (see ``_DEFAULT_DIMS``); only
            needed if a checkpoint was trained with non-default dims.

    Returns:
        A ready-to-use :class:`SemanticEvaluator`.
    """
    device = torch.device(device)
    checkpoint = Path(checkpoint)
    meta_dir = Path(meta_dir)
    glove_dir = Path(glove_dir)

    cfg = dict(_DEFAULT_DIMS)
    cfg.update(dims)

    state = torch.load(checkpoint, map_location=device)

    text_encoder = TextEncoderBiGRUCo(
        word_size=cfg["dim_word"],
        pos_size=cfg["dim_pos_ohot"],
        hidden_size=cfg["dim_text_hidden"],
        output_size=cfg["dim_coemb_hidden"],
    )
    motion_encoder = MotionEncoderBiGRUCo(
        input_size=cfg["dim_movement_latent"],
        hidden_size=cfg["dim_motion_hidden"],
        output_size=cfg["dim_coemb_hidden"],
    )
    movement_encoder = MovementConvEncoder(
        cfg["motion_input_dim"], cfg["dim_movement_enc_hidden"], cfg["dim_movement_latent"]
    )

    text_encoder.load_state_dict(state["text_encoder"])
    motion_encoder.load_state_dict(state["motion_encoder"])
    movement_encoder.load_state_dict(state["movement_encoder"])

    mean = np.load(meta_dir / "mean.npy")
    std = np.load(meta_dir / "std.npy")
    word_vectorizer = WordVectorizer(str(glove_dir), "our_vab")

    return SemanticEvaluator(
        text_encoder=text_encoder,
        motion_encoder=motion_encoder,
        movement_encoder=movement_encoder,
        word_vectorizer=word_vectorizer,
        mean=mean,
        std=std,
        device=device,
    )


if __name__ == "__main__":
    """Smoke test: build random-init encoders directly (no checkpoint needed) and
    confirm the forward/score path runs end-to-end with no absolute paths involved.
    For a real checkpoint, use load_evaluator(...) with paths under ${TSD_ASSETS}.
    """
    import tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dims = _DEFAULT_DIMS

    text_encoder = TextEncoderBiGRUCo(
        word_size=dims["dim_word"], pos_size=dims["dim_pos_ohot"],
        hidden_size=dims["dim_text_hidden"], output_size=dims["dim_coemb_hidden"],
    )
    motion_encoder = MotionEncoderBiGRUCo(
        input_size=dims["dim_movement_latent"], hidden_size=dims["dim_motion_hidden"],
        output_size=dims["dim_coemb_hidden"],
    )
    movement_encoder = MovementConvEncoder(
        dims["motion_input_dim"], dims["dim_movement_enc_hidden"], dims["dim_movement_latent"]
    )

    # Minimal fake GloVe vocab (sos/eos/unk + a couple words) so WordVectorizer works
    # without downloading the real GloVe assets.
    words = ["sos", "eos", "unk", "walk", "forward"]
    vectors = np.random.randn(len(words), dims["dim_word"]).astype(np.float32)
    word2idx = {w: i for i, w in enumerate(words)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "our_vab_data.npy", vectors)
        import pickle
        with open(tmp / "our_vab_words.pkl", "wb") as f:
            pickle.dump(words, f)
        with open(tmp / "our_vab_idx.pkl", "wb") as f:
            pickle.dump(word2idx, f)
        word_vectorizer = WordVectorizer(str(tmp), "our_vab")

    mean = np.zeros(36, dtype=np.float32)
    std = np.ones(36, dtype=np.float32)

    evaluator = SemanticEvaluator(
        text_encoder=text_encoder,
        motion_encoder=motion_encoder,
        movement_encoder=movement_encoder,
        word_vectorizer=word_vectorizer,
        mean=mean,
        std=std,
        device=device,
    )

    motion = np.random.randn(80, 36).astype(np.float32)
    caption = "the person walks forward"
    distance = evaluator.score(motion, caption)
    print(f"score(motion, caption) = {distance:.4f}")
    assert isinstance(distance, float)
    print("inference.py smoke test passed!")
