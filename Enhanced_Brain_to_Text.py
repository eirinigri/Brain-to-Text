"""
Enhanced Brain-to-Text '25 — Ultimate Pipeline
================================================
This script implements the full pipeline for lowest possible WER:

ARCHITECTURE:
  1. Pretrained GRU Decoder (from baseline checkpoint, 10% PER)
     - 5-layer GRU with patch embedding (512×14 → 768 → 41)
     - Day-specific adapters (Softsign activation)
     - Loads 44.3M pretrained parameters
  2. Conformer Encoder (our custom model, trained from scratch)
     - 4-layer Conformer with Flash Attention
     - Subject adapter (linear transform per session)

TRAINING IMPROVEMENTS:
  - Cosine annealing LR with linear warmup
  - Gradient accumulation (effective batch = 32)
  - Early stopping (patience = 7)
  - 3-fold cross-validation → ensemble
  - More aggressive augmentation

DECODING PIPELINE:
  Acoustic logits → KenLM 4-gram beam search → LLM rescoring (Llama/GPT-2)

INFERENCE (IMPROVED FOR LOWEST WER):
  - K-fold logit averaging (ensemble)
  - Test-Time Augmentation (TTA): pool candidates across multiple sigmas
  - KenLM integration via pyctcdecode
  - Competition-style candidate pooling + dedupe + selection
  - Optional LLM scoring only on top-K candidates for speed
  - Report-ready plots saved to OUTPUT_DIR

Run: python Enhanced_Brain_to_Text.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gc, copy, math, json, sys, time
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch.nn.utils.rnn as rnn_utils
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jiwer
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from sklearn.model_selection import KFold

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    # --- Paths (update these for your environment) ---
    DATA_DIR = r"/Users/eirinigriniezaki/Desktop/Brain-to-Text/t15_copyTask_neuralData/hdf5_data_final"
    PRETRAINED_CHECKPOINT = r"/Users/eirinigriniezaki/Desktop/Brain-to-Text/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint"
    OUTPUT_DIR = r"/Users/eirinigriniezaki/Desktop/Brain-to-Text/enhanced_checkpoints"
    KENLM_PATH = r"/Users/eirinigriniezaki/Desktop/Brain-to-Text/lm.arpa"

    # --- Pretrained GRU Architecture (must match checkpoint) ---
    INPUT_DIM = 512
    GRU_HIDDEN = 768
    GRU_LAYERS = 5
    GRU_DROPOUT = 0.4
    PATCH_SIZE = 14
    PATCH_STRIDE = 4
    DAY_DROPOUT = 0.2
    OUTPUT_DIM = 41      # 40 phonemes + 1 CTC blank
    N_SESSIONS = 45

    # --- Conformer Architecture ---
    CONFORMER_DIM = 256
    CONFORMER_LAYERS = 4
    CONFORMER_HEADS = 4
    CONFORMER_KERNEL = 31
    CONFORMER_DROPOUT = 0.1

    # --- Training ---
    MODE = "finetune"     # "finetune" (pretrained GRU) or "train" (Conformer from scratch)
    EPOCHS = 30
    LR = 1e-4             # Small LR for fine-tuning pretrained model
    LR_CONFORMER = 1e-3   # Larger LR for training Conformer from scratch
    LR_MIN = 1e-6
    BATCH_SIZE = 8
    GRAD_ACCUM = 4        # Effective batch = 32
    WEIGHT_DECAY = 0.01
    PATIENCE = 7
    WARMUP_EPOCHS = 3
    TRAIN_DEBUG = True     # Print target/output length diagnostics each epoch
    FAIL_FAST_ON_BAD_LOSS = True

    # --- K-Fold Ensemble ---
    N_FOLDS = 3

    # --- Signal Processing ---
    SMOOTHING_SIGMA = 2   # Baseline used sigma=2 (from args.yaml smooth_kernel_std)
    WHITE_NOISE_STD = 1.0 # From baseline augmentation
    CONSTANT_OFFSET_STD = 0.2
    TIME_MASK_PCT = 0.1
    CHANNEL_MASK_PCT = 0.1

    # --- Decoding ---
    BEAM_WIDTH = 30       # Practical for Python fallback beam search
    USE_LM = True
    USE_PYCTCDECODE = False  # Multi-character phoneme labels are incompatible with pyctcdecode token assumptions
    LM_ALPHA = 0.5        # KenLM weight
    LM_BETA = 1.5         # Word insertion bonus
    USE_LLM_RESCORE = False  # Usually improves phoneme-level WER stability
    LLM_MODEL = "meta-llama/Llama-3.2-1B"  # Llama for rescoring (can also use "gpt2" as fallback)
    LLM_FALLBACK = "gpt2"  # Fallback if Llama not available
    LLM_WEIGHT = 0.3
    N_BEST = 20
    VAL_USE_BEAM = False  # Greedy validation is faster and avoids decode-backend mismatch during training

    # --- TTA ---
    USE_TTA = True
    TTA_SIGMAS = [1.5, 1.75, 2.0, 2.25, 2.5]  # Around baseline sigma=2

    # --- Device ---
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PHONEME VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

VOCAB_LIST = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z',
    'ZH', '|'
]
TOKEN_MAP = {0: ""}  # blank
TOKEN_MAP.update({i + 1: p for i, p in enumerate(VOCAB_LIST)})
DECODER_LABELS = [""] + VOCAB_LIST

# Session list (must match checkpoint order)
SESSIONS = [
    't15.2023.08.11', 't15.2023.08.13', 't15.2023.08.18', 't15.2023.08.20',
    't15.2023.08.25', 't15.2023.08.27', 't15.2023.09.01', 't15.2023.09.03',
    't15.2023.09.24', 't15.2023.09.29', 't15.2023.10.01', 't15.2023.10.06',
    't15.2023.10.08', 't15.2023.10.13', 't15.2023.10.15', 't15.2023.10.20',
    't15.2023.10.22', 't15.2023.11.03', 't15.2023.11.04', 't15.2023.11.17',
    't15.2023.11.19', 't15.2023.11.26', 't15.2023.12.03', 't15.2023.12.08',
    't15.2023.12.10', 't15.2023.12.17', 't15.2023.12.29', 't15.2024.02.25',
    't15.2024.03.03', 't15.2024.03.08', 't15.2024.03.15', 't15.2024.03.17',
    't15.2024.04.25', 't15.2024.04.28', 't15.2024.05.10', 't15.2024.06.14',
    't15.2024.07.19', 't15.2024.07.21', 't15.2024.07.28', 't15.2025.01.10',
    't15.2025.01.12', 't15.2025.03.14', 't15.2025.03.16', 't15.2025.03.30',
    't15.2025.04.13'
]
SESSION_TO_ID = {s: i for i, s in enumerate(SESSIONS)}

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_data(data, sigma=2):
    """Gaussian smoothing matching baseline config."""
    if sigma > 0:
        return gaussian_filter1d(data, sigma=sigma, axis=0).astype(np.float32)
    return data.astype(np.float32)

def augment_neural(x_np, is_train=True):
    """Apply baseline-matching augmentation (white noise, offset, masking)."""
    if not is_train:
        return x_np

    # White noise (from baseline: white_noise_std=1.0)
    if np.random.rand() < 0.5:
        x_np = x_np + np.random.randn(*x_np.shape).astype(np.float32) * CFG.WHITE_NOISE_STD

    # Constant offset per channel (from baseline: constant_offset_std=0.2)
    if np.random.rand() < 0.5:
        offset = np.random.randn(1, x_np.shape[1]).astype(np.float32) * CFG.CONSTANT_OFFSET_STD
        x_np = x_np + offset

    # Time masking
    if np.random.rand() < 0.3:
        seq_len = x_np.shape[0]
        n_mask = int(seq_len * CFG.TIME_MASK_PCT)
        if n_mask > 0:
            start = np.random.randint(0, max(1, seq_len - n_mask))
            x_np[start:start + n_mask, :] = 0

    # Channel masking
    if np.random.rand() < 0.2:
        n_ch = x_np.shape[1]
        n_mask = int(n_ch * CFG.CHANNEL_MASK_PCT)
        if n_mask > 0:
            ch_idx = np.random.choice(n_ch, n_mask, replace=False)
            x_np[:, ch_idx] = 0

    return x_np

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class BrainDataset(Dataset):
    def __init__(self, hdf5_file, session_id, is_test=False, is_train=False,
                 smoothing_sigma=2):
        self.file_path = hdf5_file
        self.session_id = session_id
        self.is_test = is_test
        self.is_train = is_train
        self.smoothing_sigma = smoothing_sigma
        self.file = None

        try:
            with h5py.File(self.file_path, "r") as f:
                self.trial_keys = list(f.keys())
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            self.trial_keys = []

    def __len__(self):
        return len(self.trial_keys)

    def _ensure_file_open(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

    def __getitem__(self, idx):
        self._ensure_file_open()
        key = self.trial_keys[idx]
        grp = self.file[key]

        x = grp["input_features"][:].astype(np.float32)
        x = smooth_data(x, sigma=self.smoothing_sigma)
        x = augment_neural(x, is_train=self.is_train)

        # Load labels whenever available (some "test" files can still be labeled locally).
        if "seq_class_ids" in grp:
            y = grp["seq_class_ids"][:].astype(np.int64)
            # Targets are zero-padded to fixed length in this dataset; CTC targets must not include blank(0).
            y = y[y != 0]
        else:
            # Keep unlabeled samples valid for collation while signaling empty transcript.
            y = np.zeros((0,), dtype=np.int64)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            x.shape[0],
            len(y),
            self.session_id,
        )

def collate_fn(batch):
    xs, ys, xlens, ylens, sids = zip(*batch)
    xs_padded = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)
    return (
        xs_padded,
        ys_padded,
        torch.tensor(xlens, dtype=torch.long),
        torch.tensor(ylens, dtype=torch.long),
        torch.tensor(sids, dtype=torch.long),
    )

def load_split(split_name):
    """Load split (train/val/test) across all sessions."""
    datasets = []
    for session in SESSIONS:
        session_dir = os.path.join(CFG.DATA_DIR, session)
        if not os.path.exists(session_dir):
            continue

        file_name = f"data_{split_name}.hdf5"
        hdf5_path = os.path.join(session_dir, file_name)
        if not os.path.exists(hdf5_path):
            continue

        sid = SESSION_TO_ID.get(session, 0)
        ds = BrainDataset(
            hdf5_path,
            session_id=sid,
            is_test=(split_name == "test"),
            is_train=(split_name == "train"),
            smoothing_sigma=CFG.SMOOTHING_SIGMA,
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        return ConcatDataset([])
    return ConcatDataset(datasets)

# ═══════════════════════════════════════════════════════════════════════════════
# CTC DECODING (GREEDY + BEAM)
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_decode(logits):
    """Greedy CTC decode."""
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    tokens = np.argmax(logits, axis=-1)
    prev = None
    out = []
    for t in tokens:
        if t != prev and t != 0:
            out.append(TOKEN_MAP.get(int(t), ""))
        prev = t
    return " ".join(out)

def ctc_beam_search(logits, beam_width=50):
    """Simple CTC beam search (fallback if pyctcdecode unavailable)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    # Convert to log-probabilities for numerically sound beam scoring.
    logits = logits.astype(np.float64)
    logits = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

    T, C = logits.shape
    beams = {(tuple(), 0): 0.0}
    for t in range(T):
        new_beams = defaultdict(lambda: -1e9)
        for (prefix, last), score in beams.items():
            for c in range(C):
                p = logits[t, c]
                if c == 0:
                    new_beams[(prefix, last)] = np.logaddexp(new_beams[(prefix, last)], score + p)
                else:
                    if c == last:
                        new_beams[(prefix, last)] = np.logaddexp(new_beams[(prefix, last)], score + p)
                    else:
                        new_prefix = prefix + (c,)
                        new_beams[(new_prefix, c)] = np.logaddexp(new_beams[(new_prefix, c)], score + p)
        beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

    best = max(beams, key=lambda k: beams[k])
    return " ".join([TOKEN_MAP.get(i, "") for i in best[0]])

# ═══════════════════════════════════════════════════════════════════════════════
# KENLM LANGUAGE MODEL DECODER
# ═══════════════════════════════════════════════════════════════════════════════

class LMDecoder:
    """Beam search decoder with optional KenLM integration via pyctcdecode."""

    def __init__(self):
        self.decoder = None
        self._init_decoder()

    def _init_decoder(self):
        if not CFG.USE_PYCTCDECODE:
            print("  pyctcdecode disabled for phoneme-token decoding; using internal CTC beam search")
            return

        # pyctcdecode is designed for character/BPE tokens; this task uses ARPABET-like multi-char phoneme tokens.
        non_blank_labels = [l for l in DECODER_LABELS if l]
        has_multi_char_phonemes = any((len(l) > 1 and l != "|") for l in non_blank_labels)
        if has_multi_char_phonemes:
            print("  Detected multi-character phoneme labels; skipping pyctcdecode to avoid tokenization mismatch")
            return

        try:
            from pyctcdecode import build_ctcdecoder

            if CFG.USE_LM and CFG.KENLM_PATH and os.path.exists(CFG.KENLM_PATH):
                print(f"  KenLM model: {CFG.KENLM_PATH}")
                self.decoder = build_ctcdecoder(
                    labels=DECODER_LABELS,
                    kenlm_model_path=CFG.KENLM_PATH,
                    alpha=CFG.LM_ALPHA,
                    beta=CFG.LM_BETA,
                )
            else:
                # No LM, just beam search
                self.decoder = build_ctcdecoder(labels=DECODER_LABELS)
                if CFG.USE_LM and CFG.KENLM_PATH:
                    print(f"  WARNING: KenLM not found at {CFG.KENLM_PATH}")
            print("  ✓ pyctcdecode decoder ready")
        except ImportError:
            print("  pyctcdecode not installed (pip install pyctcdecode kenlm)")
            print("  Falling back to Python beam search")

    def decode(self, logits, beam_width=None):
        """Decode log-probabilities to text."""
        bw = beam_width or CFG.BEAM_WIDTH
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()

        if self.decoder is not None:
            logits64 = logits.astype(np.float64)
            logits64 = logits64 - logits64.max(axis=-1, keepdims=True)
            probs = np.exp(logits64)
            probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
            return self.decoder.decode(probs, beam_width=bw)
        else:
            return ctc_beam_search(logits, beam_width=bw)

    def decode_nbest(self, logits, beam_width=None, n_best=None):
        """Return N-best hypotheses for LLM rescoring."""
        bw = beam_width or CFG.BEAM_WIDTH
        nb = n_best or CFG.N_BEST

        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()

        if self.decoder is not None:
            try:
                logits64 = logits.astype(np.float64)
                logits64 = logits64 - logits64.max(axis=-1, keepdims=True)
                probs = np.exp(logits64)
                probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
                beams = self.decoder.decode_beams(probs, beam_width=bw)
                results = []
                for b in beams[:nb]:
                    text = b[0]
                    score = (b[3] + b[4]) if len(b) > 4 else b[3] if len(b) > 3 else 0.0
                    results.append((text, score))
                return results
            except Exception:
                return [(self.decode(logits, bw), 0.0)]
        else:
            return [(ctc_beam_search(logits, beam_width=bw), 0.0)]

# ═══════════════════════════════════════════════════════════════════════════════
# LLM RESCORER (Llama 3.2 1B / GPT-2 fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMRescorer:
    """Rescore N-best hypotheses using a pretrained LLM."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = CFG.DEVICE
        self.model_name = None
        self._init_llm()

    def _init_llm(self):
        if not CFG.USE_LLM_RESCORE:
            print("  LLM rescoring disabled")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("  transformers not installed (pip install transformers)")
            return

        # Try primary model, then fallback
        for model_name in [CFG.LLM_MODEL, CFG.LLM_FALLBACK]:
            try:
                print(f"  Loading LLM: {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                self.model.eval()

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model_name = model_name
                print(f"  ✓ LLM loaded: {model_name}")
                return
            except Exception as e:
                print(f"  Failed to load {model_name}: {str(e)[:100]}")

        print("  WARNING: No LLM available. Rescoring disabled.")
        self.model = None

    def score(self, text):
        """Compute log-likelihood of text."""
        if self.model is None or not text.strip():
            return 0.0
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            return -outputs.loss.item()  # Higher = more likely
        except Exception:
            return 0.0

    def rescore(self, hypotheses):
        """Rescore N-best list → return best text."""
        if self.model is None or not hypotheses:
            return hypotheses[0][0] if hypotheses else ""

        best_text, best_score = "", float('-inf')
        for text, beam_score in hypotheses:
            llm_score = self.score(text)
            combined = beam_score + CFG.LLM_WEIGHT * llm_score
            if combined > best_score:
                best_score = combined
                best_text = text

        return best_text

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL: PRETRAINED GRU DECODER (from baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class GRUDecoder(nn.Module):
    """
    Exact replica of the pretrained baseline GRUDecoder.
    Architecture (from checkpoint):
      day_weights:     ParameterList of 45 × [512, 512] matrices
      day_biases:      ParameterList of 45 × [1, 512] vectors
      day_layer_activation: Softsign()
      day_layer_dropout:    Dropout(0.2)
      gru:             GRU(7168, 768, num_layers=5, batch_first=True, dropout=0.4)
      out:             Linear(768, 41)
    """
    def __init__(self, n_days=CFG.N_SESSIONS):
        super().__init__()
        self.n_days = n_days

        # Day adapter (ParameterList to match checkpoint keys exactly)
        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.eye(CFG.INPUT_DIM)) for _ in range(n_days)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, CFG.INPUT_DIM)) for _ in range(n_days)
        ])
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(CFG.DAY_DROPOUT)

        # GRU with patch embedding — raw patches go straight in (no projection)
        gru_input_size = CFG.INPUT_DIM * CFG.PATCH_SIZE  # 512 * 14 = 7168
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=CFG.GRU_HIDDEN,
            num_layers=CFG.GRU_LAYERS,
            batch_first=True,
            dropout=CFG.GRU_DROPOUT,
        )
        self.out = nn.Linear(CFG.GRU_HIDDEN, CFG.OUTPUT_DIM)

    def get_output_length(self, input_len):
        """Compute output length after patching."""
        if input_len < CFG.PATCH_SIZE:
            return 1
        return 1 + (input_len - CFG.PATCH_SIZE) // CFG.PATCH_STRIDE

    def forward(self, x, session_ids):
        """
        x: [B, T, 512] neural features
        session_ids: [B] session indices
        Returns: [B, T', 41] logits (NOT log-softmax; CTC loss does its own log_softmax)
        """
        B, T, D = x.shape

        # Day adapter: per-sample linear transform
        if isinstance(session_ids, torch.Tensor):
            day_list = session_ids.tolist()
        elif isinstance(session_ids, int):
            day_list = [session_ids] * B
        else:
            day_list = list(session_ids)

        adapted = []
        for i, d in enumerate(day_list):
            d = max(0, min(d, self.n_days - 1))
            a = torch.matmul(x[i], self.day_weights[d]) + self.day_biases[d]
            adapted.append(a)
        x = torch.stack(adapted)
        x = self.day_layer_dropout(self.day_layer_activation(x))

        # Patch embedding: unfold time dimension
        if T >= CFG.PATCH_SIZE:
            patches = x.unfold(1, CFG.PATCH_SIZE, CFG.PATCH_STRIDE)  # [B, n_patches, D, patch_size]
            patches = patches.permute(0, 1, 3, 2).reshape(B, patches.size(1), -1)  # [B, n_patches, D*patch_size]
        else:
            pad_len = CFG.PATCH_SIZE - T
            x = F.pad(x, (0, 0, 0, pad_len))
            patches = x.reshape(B, 1, -1)

        # GRU
        out, _ = self.gru(patches)
        logits = self.out(out)
        return logits

def load_pretrained_gru(checkpoint_path=None):
    """Load the pretrained GRU model from the baseline checkpoint."""
    checkpoint_path = checkpoint_path or CFG.PRETRAINED_CHECKPOINT

    model = GRUDecoder(n_days=CFG.N_SESSIONS)

    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # The checkpoint might be wrapped in various ways
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Handle torch.compile wrapper keys (_orig_mod. prefix)
        clean_state = {}
        for k, v in state_dict.items():
            clean_key = k.replace("_orig_mod.", "")
            clean_state[clean_key] = v

        # Load with strict=False to handle minor key differences
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"  ✓ Loaded {len(clean_state)} parameters")
        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} total params")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("  Training GRU from scratch (this will give much worse results)")

    return model

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL: CONFORMER ENCODER (from our existing notebook)
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectAdapter(nn.Module):
    def __init__(self, in_dim=CFG.INPUT_DIM, out_dim=CFG.CONFORMER_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        out = self.dropout(out)
        return out

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.mhsa = MultiHeadSelfAttention(dim, heads=heads, dropout=dropout)
        self.conv = ConvolutionModule(dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)

class ConformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = SubjectAdapter(CFG.INPUT_DIM, CFG.CONFORMER_DIM)
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=CFG.CONFORMER_DIM,
                heads=CFG.CONFORMER_HEADS,
                kernel_size=CFG.CONFORMER_KERNEL,
                dropout=CFG.CONFORMER_DROPOUT,
            ) for _ in range(CFG.CONFORMER_LAYERS)
        ])
        self.out = nn.Linear(CFG.CONFORMER_DIM, CFG.OUTPUT_DIM)

    def forward(self, x, session_ids=None):
        x = self.adapter(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc="[Train]", leave=False)
    optimizer.zero_grad()
    accum_steps = 0
    batches_total = 0
    batches_empty_target = 0
    batches_bad_loss = 0
    tgt_len_sum, out_len_sum, sample_count = 0.0, 0.0, 0
    tgt_min, tgt_max = 10**9, -1
    out_min, out_max = 10**9, -1

    for step, batch in enumerate(pbar):
        batches_total += 1
        x, y, x_lens, y_lens, sids = batch[:5]

        # Skip samples without labels (should not happen for train/val, but fail-safe).
        valid_idx = torch.nonzero(y_lens > 0, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            batches_empty_target += 1
            continue
        if valid_idx.numel() < y_lens.numel():
            x = x.index_select(0, valid_idx)
            y = y.index_select(0, valid_idx)
            x_lens = x_lens.index_select(0, valid_idx)
            y_lens = y_lens.index_select(0, valid_idx)
            sids = sids.index_select(0, valid_idx)

        x = x.to(CFG.DEVICE)
        y = y.to(CFG.DEVICE)
        sids = sids.to(CFG.DEVICE)

        use_amp = CFG.DEVICE == "cuda"
        if use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, sids)
        else:
            logits = model(x, sids)

        # Prepare lengths for CTC
        if hasattr(model, 'get_output_length'):
            out_lens = torch.tensor([model.get_output_length(l.item()) for l in x_lens]).to(CFG.DEVICE)
        else:
            out_lens = x_lens.clone().to(CFG.DEVICE)

        out_lens = torch.clamp(out_lens, max=logits.size(1))
        y_lens = torch.minimum(y_lens.to(CFG.DEVICE), out_lens).long()

        out_l_cpu = out_lens.detach().cpu()
        y_l_cpu = y_lens.detach().cpu()
        sample_count += int(y_l_cpu.numel())
        if y_l_cpu.numel() > 0:
            tgt_len_sum += float(y_l_cpu.float().sum().item())
            out_len_sum += float(out_l_cpu.float().sum().item())
            tgt_min = min(tgt_min, int(y_l_cpu.min().item()))
            tgt_max = max(tgt_max, int(y_l_cpu.max().item()))
            out_min = min(out_min, int(out_l_cpu.min().item()))
            out_max = max(out_max, int(out_l_cpu.max().item()))

        # CTC loss expects [T,B,C]
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)

        loss = criterion(log_probs, y, out_lens, y_lens)

        if torch.isnan(loss) or torch.isinf(loss):
            batches_bad_loss += 1
            continue

        loss = loss / CFG.GRAD_ACCUM

        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_steps += 1
        if accum_steps == CFG.GRAD_ACCUM:
            if use_amp and scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if use_amp and scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            accum_steps = 0

        total_loss += loss.item() * CFG.GRAD_ACCUM
        n += 1
        pbar.set_postfix(
            loss=f"{total_loss/max(n,1):.4f}",
            bad=f"{batches_bad_loss}",
            empty=f"{batches_empty_target}",
        )

    # Flush leftover micro-batch gradients when len(loader) is not divisible by GRAD_ACCUM.
    if accum_steps > 0:
        if CFG.DEVICE == "cuda" and scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if CFG.DEVICE == "cuda" and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if CFG.TRAIN_DEBUG:
        avg_tgt = (tgt_len_sum / sample_count) if sample_count > 0 else 0.0
        avg_out = (out_len_sum / sample_count) if sample_count > 0 else 0.0
        print(
            f"    [TrainDiag] batches={batches_total} used={n} bad_loss={batches_bad_loss} "
            f"empty_target={batches_empty_target} y_len(avg/min/max)={avg_tgt:.1f}/{(tgt_min if tgt_max >= 0 else 0)}/{(tgt_max if tgt_max >= 0 else 0)} "
            f"out_len(avg/min/max)={avg_out:.1f}/{(out_min if out_max >= 0 else 0)}/{(out_max if out_max >= 0 else 0)}"
        )

    if n == 0:
        raise RuntimeError(
            "No valid training batches produced finite CTC loss. "
            "Check target preprocessing and sequence lengths."
        )
    if CFG.FAIL_FAST_ON_BAD_LOSS and batches_bad_loss > max(10, int(0.3 * max(1, batches_total))):
        raise RuntimeError(
            f"Too many invalid training losses: {batches_bad_loss}/{batches_total}. "
            "Likely target/output length mismatch or malformed labels."
        )

    return total_loss / max(n, 1)

def validate_epoch(model, loader, criterion, lm_decoder=None, use_beam=False):
    """Validate model and compute WER."""
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    batches_total = 0
    unlabeled_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            batches_total += 1
            x, y, x_lens, y_lens, sids = batch[:5]
            x = x.to(CFG.DEVICE)
            sids = sids.to(CFG.DEVICE)

            use_amp = CFG.DEVICE == "cuda"
            if use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x, sids)
            else:
                logits = model(x, sids)

            # Loss: only on labeled samples (y_len > 0)
            labeled_idx = torch.nonzero(y_lens > 0, as_tuple=False).squeeze(1)
            unlabeled_samples += int(y_lens.numel() - labeled_idx.numel())
            if labeled_idx.numel() > 0:
                logits_l = logits.index_select(0, labeled_idx.to(logits.device))
                y_l = y.index_select(0, labeled_idx)
                x_lens_l = x_lens.index_select(0, labeled_idx)
                y_lens_l = y_lens.index_select(0, labeled_idx)

                if hasattr(model, 'get_output_length'):
                    out_lens = torch.tensor([model.get_output_length(l.item()) for l in x_lens_l])
                else:
                    out_lens = x_lens_l.clone()
                out_lens = torch.clamp(out_lens, max=logits_l.size(1))
                y_lens_c = torch.minimum(y_lens_l, out_lens).long()
                loss = criterion(logits_l.log_softmax(-1).permute(1, 0, 2).float().cpu(), y_l, out_lens, y_lens_c)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item() * logits_l.size(0)
                    n += logits_l.size(0)

            # Decode
            logits_cpu = logits.float().cpu()
            for i in range(x.size(0)):
                if y_lens[i].item() <= 0:
                    continue
                if hasattr(model, 'get_output_length'):
                    ol = min(logits.size(1), int(model.get_output_length(x_lens[i].item())))
                else:
                    ol = x_lens[i].item()
                pred_logits = logits_cpu[i, :ol, :]

                if use_beam and lm_decoder:
                    pred = lm_decoder.decode(pred_logits)
                elif use_beam:
                    pred = ctc_beam_search(pred_logits, beam_width=CFG.BEAM_WIDTH)
                else:
                    pred = greedy_decode(pred_logits)

                true = " ".join([TOKEN_MAP.get(idx.item(), "") for idx in y[i, :y_lens[i]]])
                all_pred.append(normalize_phoneme_text(pred))
                all_true.append(normalize_phoneme_text(true))

    if CFG.TRAIN_DEBUG:
        print(
            f"    [ValDiag] batches={batches_total} labeled={n} unlabeled={unlabeled_samples} "
            f"pairs_for_wer={len(all_true)}"
        )

    wer = jiwer.wer(all_true, all_pred) if all_true else 1.0
    return total_loss / max(n, 1), wer

# ═══════════════════════════════════════════════════════════════════════════════
# TTA
# ═══════════════════════════════════════════════════════════════════════════════

def tta_predict(model, x_numpy, session_id, sigmas=None):
    """Average predictions over multiple smoothing parameters."""
    sigmas = sigmas or CFG.TTA_SIGMAS
    model.eval()
    all_logits = []

    for sigma in sigmas:
        x_s = smooth_data(x_numpy, sigma=sigma)
        xt = torch.tensor(x_s, dtype=torch.float32).unsqueeze(0).to(CFG.DEVICE)
        sid = torch.tensor([session_id], dtype=torch.long).to(CFG.DEVICE)

        with torch.no_grad():
            if CFG.DEVICE == "cuda":
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(xt, sid)
            else:
                logits = model(xt, sid)
        all_logits.append(logits.float().cpu())

    return torch.stack(all_logits).mean(0)[0]  # [T, C]

def _average_variable_length_logits(logits_list):
    """Average a list of [T_i, C] logits without padding bias."""
    if not logits_list:
        return None
    if len(logits_list) == 1:
        return logits_list[0]

    max_len = max(l.size(0) for l in logits_list)
    n_classes = logits_list[0].size(1)
    device = logits_list[0].device
    dtype = logits_list[0].dtype

    sum_logits = torch.zeros((max_len, n_classes), dtype=dtype, device=device)
    counts = torch.zeros((max_len, 1), dtype=dtype, device=device)
    for l in logits_list:
        L = l.size(0)
        sum_logits[:L] += l
        counts[:L] += 1.0

    return sum_logits / torch.clamp(counts, min=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION + CANDIDATE SELECTION (WER-ORIENTED)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_phoneme_text(s: str) -> str:
    """Conservative normalization to avoid 'cheap' WER penalties.

    Assumes the transcript is a space-separated phoneme sequence with optional '|' token.
    We intentionally *do not* remove tokens, only normalize whitespace and casing.
    """
    if s is None:
        return ""
    s = str(s).strip().upper()
    # Collapse whitespace
    s = " ".join(s.split())
    # Normalize spacing around '|'
    s = s.replace(" | ", " | ").replace("| ", "| ").replace(" |", " |")
    # Ensure single-spaced tokens
    s = " ".join(s.split())
    # Collapse consecutive separators
    toks = s.split(" ")
    out = []
    prev = None
    for t in toks:
        if t == "" or t is None:
            continue
        if t == "|" and prev == "|":
            continue
        out.append(t)
        prev = t
    return " ".join(out).strip()

def dedupe_hypotheses(hyps, keep_top=200):
    """Deduplicate hypotheses by normalized text, keep best (max score)."""
    best = {}
    for txt, score in hyps:
        n = normalize_phoneme_text(txt)
        if not n:
            continue
        if n not in best or score > best[n]:
            best[n] = score
    items = sorted(best.items(), key=lambda x: x[1], reverse=True)[:keep_top]
    return [(t, s) for t, s in items]

class CandidateSelector:
    """Competition-style selector: combine beam/LM scores with optional LLM scoring."""

    def __init__(self, llm_rescorer=None):
        self.llm = llm_rescorer

    def select(self, hyps, topk_llm=30):
        if not hyps:
            return ""
        hyps = dedupe_hypotheses(hyps, keep_top=max(topk_llm, 50))

        # If no LLM, return best beam score
        if self.llm is None or self.llm.model is None or not CFG.USE_LLM_RESCORE:
            return hyps[0][0]

        # Only score top-k with LLM for speed
        scored = []
        for i, (txt, beam_score) in enumerate(hyps):
            if i < topk_llm:
                llm_score = self.llm.score(txt)
                combined = beam_score + CFG.LLM_WEIGHT * llm_score
            else:
                combined = beam_score
            scored.append((txt, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PLOTS (saved to OUTPUT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

def _per_utterance_wer(truths, preds):
    """Return list of per-utterance WER values."""
    wers = []
    for t, p in zip(truths, preds):
        t_n = normalize_phoneme_text(t)
        p_n = normalize_phoneme_text(p)
        try:
            w = jiwer.wer([t_n], [p_n])
        except Exception:
            w = 1.0
        wers.append(float(w))
    return wers

def _align_tokens(ref_tokens, hyp_tokens):
    """Levenshtein alignment on token sequences."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=diag, 1=up(del), 2=left(ins)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = 2

    for i in range(1, n + 1):
        r = ref_tokens[i - 1]
        for j in range(1, m + 1):
            h = hyp_tokens[j - 1]
            c_diag = dp[i - 1, j - 1] + (0 if r == h else 1)
            c_up = dp[i - 1, j] + 1
            c_left = dp[i, j - 1] + 1

            best = c_diag
            op = 0
            if c_up < best:
                best = c_up
                op = 1
            if c_left < best:
                best = c_left
                op = 2

            dp[i, j] = best
            bt[i, j] = op

    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i, j] == 0:
            r = ref_tokens[i - 1]
            h = hyp_tokens[j - 1]
            if r == h:
                aligned.append(("eq", r, h))
            else:
                aligned.append(("sub", r, h))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bt[i, j] == 1):
            aligned.append(("del", ref_tokens[i - 1], None))
            i -= 1
        else:
            aligned.append(("ins", None, hyp_tokens[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned

def _phoneme_substitution_matrix(truths, preds, top_k=18):
    """Build substitution confusion matrix for top phonemes."""
    subs = defaultdict(int)
    token_mass = defaultdict(int)

    for t, p in zip(truths, preds):
        t_tokens = normalize_phoneme_text(t).split()
        p_tokens = normalize_phoneme_text(p).split()
        for op, ref_t, hyp_t in _align_tokens(t_tokens, p_tokens):
            if op == "sub" and ref_t and hyp_t:
                subs[(ref_t, hyp_t)] += 1
                token_mass[ref_t] += 1
                token_mass[hyp_t] += 1

    if not subs:
        return None, None

    tokens = [t for t, _ in sorted(token_mass.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    tok_to_idx = {t: i for i, t in enumerate(tokens)}
    mat = np.zeros((len(tokens), len(tokens)), dtype=np.float32)
    for (r, h), cnt in subs.items():
        if r in tok_to_idx and h in tok_to_idx:
            mat[tok_to_idx[r], tok_to_idx[h]] += float(cnt)

    return mat, tokens

def save_report_plots(truths, preds, out_dir):
    """Create plots you can drop into the report."""
    os.makedirs(out_dir, exist_ok=True)

    # Token lengths
    t_lens = [len(normalize_phoneme_text(t).split()) for t in truths]
    p_lens = [len(normalize_phoneme_text(p).split()) for p in preds]
    wers = _per_utterance_wer(truths, preds)

    # 1) Length scatter (true vs pred)
    plt.figure(figsize=(6, 5))
    plt.scatter(t_lens, p_lens, s=12, alpha=0.6)
    mx = max(t_lens + p_lens + [1])
    plt.plot([0, mx], [0, mx], linewidth=1)
    plt.xlabel("True length (tokens)")
    plt.ylabel("Predicted length (tokens)")
    plt.title("Length sanity check: true vs predicted")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_length_scatter.png"), dpi=200)
    plt.close()

    # 2) Length histograms
    plt.figure(figsize=(8, 4))
    plt.hist(t_lens, bins=30, alpha=0.6, label="True")
    plt.hist(p_lens, bins=30, alpha=0.6, label="Pred")
    plt.xlabel("Length (tokens)")
    plt.ylabel("Count")
    plt.title("Length distribution")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "length_histograms.png"), dpi=200)
    plt.close()

    # 3) WER vs length (binned)
    # Bin by true length
    bins = [0, 10, 20, 30, 40, 60, 80, 120, 2000]
    bin_labels = []
    bin_wers = []
    for a, b in zip(bins[:-1], bins[1:]):
        idx = [i for i, L in enumerate(t_lens) if a <= L < b]
        if not idx:
            continue
        bin_labels.append(f"{a}-{b-1}")
        bin_wers.append(float(np.mean([wers[i] for i in idx])) * 100)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(bin_wers)), bin_wers, marker='o')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=30, ha='right')
    plt.ylabel("WER (%)")
    plt.xlabel("True length bin (tokens)")
    plt.title("WER by utterance length")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wer_by_length.png"), dpi=200)
    plt.close()

    # 4) Phoneme substitution confusion matrix
    mat, toks = _phoneme_substitution_matrix(truths, preds, top_k=18)
    if mat is not None and mat.size > 0 and np.any(mat > 0):
        plt.figure(figsize=(10, 8))
        im = plt.imshow(np.log1p(mat), interpolation="nearest", cmap="magma", aspect="auto")
        plt.colorbar(im, label="log(1 + substitution count)")
        plt.xticks(range(len(toks)), toks, rotation=90)
        plt.yticks(range(len(toks)), toks)
        plt.xlabel("Predicted phoneme")
        plt.ylabel("True phoneme")
        plt.title("Top phoneme substitutions (confusion matrix)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phoneme_confusion_matrix.png"), dpi=220)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Full training + inference pipeline."""

    print("\n" + "═" * 70)
    print("  ENHANCED BRAIN-TO-TEXT PIPELINE")
    print("═" * 70)
    print(f"  Device:      {CFG.DEVICE}")
    print(f"  Mode:        {CFG.MODE}")
    print(f"  Epochs:      {CFG.EPOCHS} (patience={CFG.PATIENCE})")
    print(f"  Batch:       {CFG.BATCH_SIZE} × {CFG.GRAD_ACCUM} = {CFG.BATCH_SIZE * CFG.GRAD_ACCUM}")
    print(f"  K-Folds:     {CFG.N_FOLDS}")
    print(f"  Beam Width:  {CFG.BEAM_WIDTH}")
    print(f"  Val decode:  {'beam' if CFG.VAL_USE_BEAM else 'greedy'}")
    print(f"  TTA:         {CFG.USE_TTA} ({CFG.TTA_SIGMAS})")
    print(f"  LLM:         {CFG.LLM_MODEL}")

    # --- Load data ---
    print("\n[1/5] Loading data...")
    train_ds = load_split('train')
    val_ds = load_split('val')
    test_ds = load_split('test')
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # --- Initialize decoders ---
    print("\n[2/5] Initializing decoders...")
    lm_decoder = LMDecoder()
    llm_rescorer = LLMRescorer()

    # --- K-Fold Training ---
    print(f"\n[3/5] Training ({CFG.N_FOLDS}-fold)...")
    combined_ds = ConcatDataset([train_ds, val_ds])
    kfold = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_states = []
    fold_wers = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(combined_ds)))):
        print(f"\n{'─' * 50}")
        print(f"  FOLD {fold + 1}/{CFG.N_FOLDS}")
        print(f"{'─' * 50}")

        train_loader = DataLoader(Subset(combined_ds, train_idx.tolist()),
                                   batch_size=CFG.BATCH_SIZE, shuffle=True,
                                   collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(Subset(combined_ds, val_idx.tolist()),
                                 batch_size=CFG.BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)

        # Create model
        if CFG.MODE == "finetune":
            model = load_pretrained_gru()
            lr = CFG.LR   # Small LR for fine-tuning
        else:
            model = ConformerModel()
            lr = CFG.LR_CONFORMER

        model = model.to(CFG.DEVICE)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.WEIGHT_DECAY)
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        scaler = GradScaler('cuda') if CFG.DEVICE == "cuda" else None

        best_wer, best_state = float('inf'), None
        patience_ctr = 0
        history = {'train_loss': [], 'val_loss': [], 'wer': [], 'lr': []}

        for epoch in range(CFG.EPOCHS):
            # Warmup + cosine
            if epoch < CFG.WARMUP_EPOCHS:
                cur_lr = lr * (epoch + 1) / CFG.WARMUP_EPOCHS
            else:
                cur_lr = cosine_lr(epoch - CFG.WARMUP_EPOCHS, max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS), lr, CFG.LR_MIN)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, wer = validate_epoch(
                model,
                val_loader,
                criterion,
                lm_decoder=lm_decoder,
                use_beam=CFG.VAL_USE_BEAM,
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['wer'].append(wer)
            history['lr'].append(cur_lr)

            print(f"    Epoch {epoch+1:02d}: train={train_loss:.4f} val={val_loss:.4f} WER={wer*100:.2f}% lr={cur_lr:.2e}")

            # Early stopping on WER
            if wer < best_wer:
                best_wer = wer
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= CFG.PATIENCE:
                    print(f"       → Early stopping (patience={CFG.PATIENCE})")
                    break

        fold_states.append(best_state)
        fold_wers.append(best_wer)

        # Plot
        _plot_history(history, fold)

        del model, optimizer, scaler
        if CFG.DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  K-Fold Results:")
    for i, w in enumerate(fold_wers):
        print(f"    Fold {i+1}: WER = {w*100:.2f}%")
    print(f"    Avg:    WER = {np.mean(fold_wers)*100:.2f}%")

    # --- Ensemble Inference ---
    print(f"\n[4/5] Ensemble Inference...")
    final_wer = _ensemble_inference(fold_states, test_ds, lm_decoder, llm_rescorer)

    # --- Summary ---
    print(f"\n[5/5] Summary")
    print("═" * 70)
    print(f"  FINAL ENSEMBLE WER: {final_wer*100:.2f}%")
    print(f"  Models:     {len(fold_states)} folds ({'pretrained GRU' if CFG.MODE == 'finetune' else 'Conformer'})")
    print(f"  KenLM:      {'Active' if (lm_decoder.decoder is not None and CFG.KENLM_PATH) else 'Not configured'}")
    print(f"  LLM:        {llm_rescorer.model_name or 'Not available'}")
    print(f"  TTA:        {'Active' if CFG.USE_TTA else 'Off'}")
    print("═" * 70)

    return final_wer


def _ensemble_inference(fold_states, test_ds, lm_decoder, llm_rescorer):
    """Run ensemble inference with competition-style candidate pooling + selection.

    Key improvements vs the original:
      - Pool N-best candidates across multiple TTA smoothing sigmas
      - Average logits across folds *per sigma* (logit-level ensemble)
      - Deduplicate hypotheses and pick the best using beam/LM score + optional LLM score
      - Save report-ready plots at the end
    """
    # Load all fold models
    models = []
    for state in fold_states:
        if CFG.MODE == "finetune":
            m = GRUDecoder(n_days=CFG.N_SESSIONS)
        else:
            m = ConformerModel()
        m.load_state_dict(state)
        m = m.to(CFG.DEVICE)
        m.eval()
        models.append(m)

    selector = CandidateSelector(llm_rescorer)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    all_pred, all_true = [], []
    eval_pred, eval_true = [], []
    used_sigmas = CFG.TTA_SIGMAS if CFG.USE_TTA else [CFG.SMOOTHING_SIGMA]

    for batch in tqdm(test_loader, desc="Ensemble Inference"):
        x, y, x_lens, y_lens, sids = batch[:5]
        session_id = int(sids[0].item())
        x_np = x[0].numpy()

        true = ""
        has_label = int(y_lens[0].item()) > 0
        if has_label:
            true = " ".join([TOKEN_MAP.get(idx.item(), "") for idx in y[0, :y_lens[0]]])
            true = normalize_phoneme_text(true)
        all_true.append(true)

        pooled_hyps = []

        # --- Per-sigma logit ensemble (fold average) ---
        sigma_logits_list = []
        for sigma in used_sigmas:
            fold_logits = []
            x_s = smooth_data(x_np, sigma=sigma)

            for model in models:
                xt = torch.tensor(x_s, dtype=torch.float32).unsqueeze(0).to(CFG.DEVICE)
                sid = torch.tensor([session_id], dtype=torch.long).to(CFG.DEVICE)

                with torch.no_grad():
                    if CFG.DEVICE == "cuda":
                        with autocast('cuda', dtype=torch.bfloat16):
                            logits = model(xt, sid)[0].float().cpu()
                    else:
                        logits = model(xt, sid)[0].float().cpu()

                # Clip to output length
                if hasattr(model, 'get_output_length'):
                    ol = model.get_output_length(x_lens[0].item())
                else:
                    ol = x_lens[0].item()
                ol = min(int(ol), logits.size(0))
                fold_logits.append(logits[:ol])

            # Average across folds without padding bias.
            avg_logits = _average_variable_length_logits(fold_logits)
            sigma_logits_list.append(avg_logits)

            # Decode N-best for this sigma and add to pool
            if lm_decoder is not None:
                hyps = lm_decoder.decode_nbest(avg_logits, n_best=max(CFG.N_BEST, 30))
                pooled_hyps.extend(hyps)
            else:
                pooled_hyps.append((ctc_beam_search(avg_logits, beam_width=CFG.BEAM_WIDTH), 0.0))

        # --- Also add a "sigma-averaged" logit ensemble candidate set ---
        if len(sigma_logits_list) > 1:
            avg_over_sigma = _average_variable_length_logits(sigma_logits_list)
            if lm_decoder is not None:
                hyps = lm_decoder.decode_nbest(avg_over_sigma, n_best=max(CFG.N_BEST, 50))
                pooled_hyps.extend(hyps)
            else:
                pooled_hyps.append((ctc_beam_search(avg_over_sigma, beam_width=CFG.BEAM_WIDTH), 0.0))

        # --- Final selection ---
        pred = selector.select(pooled_hyps, topk_llm=30)
        pred = normalize_phoneme_text(pred)
        all_pred.append(pred)
        if has_label:
            eval_true.append(true)
            eval_pred.append(pred)

    if eval_true:
        wer = jiwer.wer(eval_true, eval_pred)
    else:
        wer = float('nan')
        print("  WARNING: No labeled samples found in inference split; WER is undefined.")

    # Save report plots
    try:
        if eval_true:
            save_report_plots(eval_true, eval_pred, CFG.OUTPUT_DIR)
        # Also save a CSV with predictions for easy inspection
        pd.DataFrame(
            {
                "true": all_true,
                "pred": all_pred,
                "has_label": [int(t != "") for t in all_true],
            }
        ).to_csv(
            os.path.join(CFG.OUTPUT_DIR, "inference_predictions.csv"), index=False
        )
    except Exception as e:
        print(f"  WARNING: Could not save plots/CSV: {e}")

    # Show examples
    print(f"\n  Sample predictions:")
    for i in range(min(5, len(all_pred))):
        print(f"    TRUE: {all_true[i][:120]}")
        print(f"    PRED: {all_pred[i][:120]}")
        print()

    del models
    if CFG.DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return wer


def _plot_history(history, fold):
    """Save training history plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r--', label='Val')
    axes[0].set_title(f'Fold {fold+1} Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [w*100 for w in history['wer']], 'g-o', ms=3)
    axes[1].set_title(f'Fold {fold+1} WER (%)')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(epochs, history['lr'], 'm-')
    axes[2].set_title(f'Fold {fold+1} LR')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, f'fold_{fold}_history.png'), dpi=150)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DOMAIN-SPECIFIC KENLM FROM TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def extract_corpus_and_build_lm():
    """
    Extract all transcription text from training data and build a KenLM model.
    Requires: pip install pyctcdecode kenlm
    And kenlm/lmplz binary in PATH (install from https://github.com/kpu/kenlm)
    """
    import subprocess

    print("\n[LM] Extracting training transcriptions...")
    corpus_path = os.path.join(CFG.OUTPUT_DIR, "training_corpus.txt")
    arpa_path = os.path.join(CFG.OUTPUT_DIR, "4gram.arpa")

    # Extract all phoneme sequences from training data
    all_texts = []
    for session in SESSIONS:
        train_file = os.path.join(CFG.DATA_DIR, session, "data_train.hdf5")
        if not os.path.exists(train_file):
            continue
        with h5py.File(train_file, 'r') as f:
            for k in f.keys():
                if 'seq_class_ids' in f[k]:
                    ids = f[k]['seq_class_ids'][:]
                    text = " ".join([TOKEN_MAP.get(int(i), "") for i in ids if i != 0])
                    if text.strip():
                        all_texts.append(text)

    # Write corpus
    with open(corpus_path, 'w') as f:
        for t in all_texts:
            f.write(t + "\n")
    print(f"  Wrote {len(all_texts)} sentences to {corpus_path}")

    # Build KenLM (requires lmplz binary)
    try:
        result = subprocess.run(
            f"lmplz -o 4 < {corpus_path} > {arpa_path}",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Built 4-gram LM: {arpa_path}")
            CFG.KENLM_PATH = arpa_path
        else:
            print(f"  lmplz failed: {result.stderr[:200]}")
            print("  Install KenLM: https://github.com/kpu/kenlm")
    except FileNotFoundError:
        print("  lmplz not found. Install KenLM to build domain-specific LM.")
        print("  The corpus has been saved. You can build the LM manually:")
        print(f"    lmplz -o 4 < {corpus_path} > {arpa_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Optionally build domain-specific LM first
    if "--build-lm" in sys.argv:
        extract_corpus_and_build_lm()

    final_wer = run_pipeline()
    print(f"\n✅ Done! Final WER: {final_wer*100:.2f}%")
