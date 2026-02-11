#!/usr/bin/env python3
"""Train a BiLSTM diacritizer for Yorùbá.

This script trains a character-level BiLSTM to restore diacritics
(both dot-below and tonal marks) to undiacritized Yorùbá text.

Usage:
    # On Google Colab (recommended):
    !pip install datasets torch
    !python train_lstm_diacritizer.py

    # Locally (CPU, takes 6-12 hours):
    python train_lstm_diacritizer.py --device cpu

    # Locally with GPU:
    python train_lstm_diacritizer.py --device cuda

The trained model will be saved to `yoruba_lstm_diacritizer.pt` (~10-15MB).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Character Vocabulary
# =============================================================================

class CharVocab:
    """Character vocabulary for encoding/decoding text."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self):
        self.char2idx: Dict[str, int] = {
            self.PAD: 0,
            self.UNK: 1,
            self.SOS: 2,
            self.EOS: 3,
        }
        self.idx2char: Dict[int, str] = {v: k for k, v in self.char2idx.items()}
        self.frozen = False

    def add_char(self, char: str) -> int:
        if char not in self.char2idx:
            if self.frozen:
                return self.char2idx[self.UNK]
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        return self.char2idx[char]

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, self.char2idx[self.UNK]) for c in text]

    def decode(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            if idx == self.char2idx[self.EOS]:
                break
            if idx not in (self.char2idx[self.PAD], self.char2idx[self.SOS]):
                chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.char2idx)

    def freeze(self):
        self.frozen = True

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(v): k for k, v in vocab.char2idx.items()}
        vocab.frozen = True
        return vocab


# =============================================================================
# Dataset
# =============================================================================

class DiacritizationDataset(Dataset):
    """Dataset for diacritization training."""

    def __init__(
        self,
        undiacritized: List[str],
        diacritized: List[str],
        vocab: CharVocab,
    ):
        self.undiacritized = undiacritized
        self.diacritized = diacritized
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.undiacritized)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.undiacritized[idx]
        tgt = self.diacritized[idx]

        src_ids = torch.tensor(self.vocab.encode(src), dtype=torch.long)
        tgt_ids = torch.tensor(self.vocab.encode(tgt), dtype=torch.long)

        return src_ids, tgt_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function with padding."""
    srcs, tgts = zip(*batch)

    src_lens = torch.tensor([len(s) for s in srcs])

    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=0)

    return srcs_padded, tgts_padded, src_lens


# =============================================================================
# Model
# =============================================================================

class BiLSTMDiacritizer(nn.Module):
    """Bidirectional LSTM for character-level diacritization.

    Architecture:
        Input characters → Embedding → BiLSTM → Linear → Output characters

    The model predicts the diacritized character for each input character,
    making it a sequence labeling task (not seq2seq generation).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len]

        Returns:
            Output logits of shape [batch, seq_len, vocab_size]
        """
        # Embed: [batch, seq, embed_dim]
        emb = self.embedding(x)
        emb = self.dropout(emb)

        # LSTM: [batch, seq, hidden_dim * 2]
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)

        # Project: [batch, seq, vocab_size]
        logits = self.fc(lstm_out)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict output character indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: BiLSTMDiacritizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt, src_lens) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(src)

        # Reshape for loss: [batch * seq, vocab] vs [batch * seq]
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_flat = tgt.view(-1)

        loss = criterion(logits_flat, tgt_flat)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 500 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(
    model: BiLSTMDiacritizer,
    dataloader: DataLoader,
    criterion: nn.Module,
    vocab: CharVocab,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    with torch.no_grad():
        for src, tgt, src_lens in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src)

            # Loss
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = tgt.view(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=-1)

            for i in range(len(src)):
                length = src_lens[i].item()
                pred_seq = preds[i, :length].cpu().tolist()
                tgt_seq = tgt[i, :length].cpu().tolist()

                # Character accuracy
                for p, t in zip(pred_seq, tgt_seq):
                    if t != 0:  # Ignore padding
                        total_chars += 1
                        if p == t:
                            correct_chars += 1

                # Word accuracy (decode and compare)
                pred_text = vocab.decode(pred_seq)
                tgt_text = vocab.decode(tgt_seq)

                pred_words = pred_text.split()
                tgt_words = tgt_text.split()

                for pw, tw in zip(pred_words, tgt_words):
                    total_words += 1
                    if pw == tw:
                        correct_words += 1

    avg_loss = total_loss / len(dataloader)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    word_acc = correct_words / total_words if total_words > 0 else 0

    return avg_loss, char_acc, word_acc


# =============================================================================
# Main
# =============================================================================

def load_data(max_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load training data from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading dataset from HuggingFace...")
    ds = load_dataset("bumie-e/Yoruba-diacritics-vs-non-diacritics", split="train")

    undiacritized = []
    diacritized = []

    for item in ds:
        undiac = item.get("no_diacritcs", "")
        diac = item.get("diacritcs", "")

        if undiac and diac and len(undiac) < 200:  # Skip very long sentences
            undiacritized.append(undiac)
            diacritized.append(diac)

        if max_samples and len(undiacritized) >= max_samples:
            break

    logger.info(f"Loaded {len(undiacritized)} sentence pairs")
    return undiacritized, diacritized


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM Yorùbá diacritizer")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples (for testing)")
    parser.add_argument("--output", type=str, default="yoruba_lstm_diacritizer.pt")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load data
    undiacritized, diacritized = load_data(args.max_samples)

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = CharVocab()
    for text in undiacritized + diacritized:
        for char in text:
            vocab.add_char(char)
    vocab.freeze()
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Train/val split
    random.seed(42)
    indices = list(range(len(undiacritized)))
    random.shuffle(indices)

    split = int(0.95 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_undiac = [undiacritized[i] for i in train_indices]
    train_diac = [diacritized[i] for i in train_indices]
    val_undiac = [undiacritized[i] for i in val_indices]
    val_diac = [diacritized[i] for i in val_indices]

    logger.info(f"Train: {len(train_undiac)}, Val: {len(val_undiac)}")

    # Datasets
    train_dataset = DiacritizationDataset(train_undiac, train_diac, vocab)
    val_dataset = DiacritizationDataset(val_undiac, val_diac, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = BiLSTMDiacritizer(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    best_word_acc = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, char_acc, word_acc = evaluate(model, val_loader, criterion, vocab, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Char Acc: {char_acc:.2%}, Word Acc: {word_acc:.2%}"
        )

        # Save best model
        if word_acc > best_word_acc:
            best_word_acc = word_acc

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": vocab.char2idx,
                "config": {
                    "vocab_size": len(vocab),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                "metrics": {
                    "char_acc": char_acc,
                    "word_acc": word_acc,
                },
            }
            torch.save(checkpoint, args.output)
            logger.info(f"  Saved best model (word_acc: {word_acc:.2%})")

    total_time = time.time() - start_time
    logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
    logger.info(f"Best word accuracy: {best_word_acc:.2%}")
    logger.info(f"Model saved to: {args.output}")

    # Model size
    model_size = os.path.getsize(args.output) / (1024 * 1024)
    logger.info(f"Model size: {model_size:.1f} MB")


if __name__ == "__main__":
    main()
