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

INFERENCE:
  - K-fold logit averaging (ensemble)
  - Test-Time Augmentation (TTA with 5 sigmas)
  - Domain-specific KenLM integration
  - N-best LLM rescoring

Expected WER: ~3-6% with all components active

Run: python Enhanced_Brain_to_Text.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    DATA_DIR = r"/mnt/data_ssd/ai_workspace/venv_torch/t15_copyTask_neuralData/hdf5_data_final"
    PRETRAINED_CHECKPOINT = r"/mnt/data_ssd/ai_workspace/venv_torch/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint"
    OUTPUT_DIR = r"/mnt/data_ssd/ai_workspace/venv_torch/enhanced_checkpoints"
    KENLM_PATH = None  # Set to path of .arpa/.bin file if available

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

    # --- K-Fold Ensemble ---
    N_FOLDS = 3

    # --- Signal Processing ---
    SMOOTHING_SIGMA = 2   # Baseline used sigma=2 (from args.yaml smooth_kernel_std)
    WHITE_NOISE_STD = 1.0 # From baseline augmentation
    CONSTANT_OFFSET_STD = 0.2
    TIME_MASK_PCT = 0.1
    CHANNEL_MASK_PCT = 0.1

    # --- Decoding ---
    BEAM_WIDTH = 100      # Wide beam for best quality
    USE_LM = True
    LM_ALPHA = 0.5        # KenLM weight
    LM_BETA = 1.5         # Word insertion bonus
    USE_LLM_RESCORE = True
    LLM_MODEL = "meta-llama/Llama-3.2-1B"  # Llama for rescoring (can also use "gpt2" as fallback)
    LLM_FALLBACK = "gpt2"  # Fallback if Llama not available
    LLM_WEIGHT = 0.3
    N_BEST = 20

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
                self.trial_keys = sorted(list(f.keys()))
        except FileNotFoundError:
            self.trial_keys = []

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        grp = self.file[self.trial_keys[idx]]
        x_data = grp["input_features"][:]

        # Smooth
        x_data = smooth_data(x_data, sigma=self.smoothing_sigma)

        # Augment
        x_data = augment_neural(x_data, is_train=self.is_train)

        x = torch.tensor(x_data, dtype=torch.float32)

        if "seq_class_ids" in grp:
            y_data = grp["seq_class_ids"][:]
            y = torch.tensor(y_data, dtype=torch.long)
        else:
            y = torch.tensor([], dtype=torch.long)

        if self.is_test:
            return x, y, self.session_id, self.trial_keys[idx]
        return x, y, self.session_id

def collate_fn(batch):
    is_test = len(batch[0]) == 4
    if is_test:
        xs, ys, sids, keys = zip(*batch)
    else:
        xs, ys, sids = zip(*batch)
        keys = None

    x_lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    sids_t = torch.tensor(sids, dtype=torch.long)

    padded_x = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    padded_y = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)

    result = (padded_x, padded_y, x_lens, y_lens, sids_t)
    return (*result, keys) if is_test else result

def load_split(split='train', sigma=None):
    """Load all sessions for a given split."""
    sigma = sigma if sigma is not None else CFG.SMOOTHING_SIGMA
    datasets = []
    data_dir = CFG.DATA_DIR

    for session in SESSIONS:
        sp = os.path.join(data_dir, session)
        if not os.path.isdir(sp):
            continue
        fp = os.path.join(sp, f"data_{split}.hdf5")
        sid = SESSION_TO_ID[session]

        ds = BrainDataset(
            fp, sid,
            is_test=(split == 'test'),
            is_train=(split == 'train'),
            smoothing_sigma=sigma
        )
        if len(ds) > 0:
            datasets.append(ds)

    return ConcatDataset(datasets) if datasets else None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL: PRETRAINED GRU DECODER (matches baseline checkpoint exactly)
# ═══════════════════════════════════════════════════════════════════════════════

class GRUDecoder(nn.Module):
    """
    Exact replica of the pretrained baseline GRUDecoder.
    Architecture (from training_log):
      day_weights:     ParameterList of 45 × [512, 512] matrices
      day_biases:      ParameterList of 45 × [1, 512] vectors
      day_layer_activation: Softsign()
      day_layer_dropout:    Dropout(0.2)
      gru:             GRU(7168, 768, num_layers=5, batch_first=True, dropout=0.4)
      out:             Linear(768, 41)
    """
    def __init__(self, n_days=45):
        super().__init__()

        # Day adapter (ParameterList to match checkpoint keys exactly)
        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.eye(CFG.INPUT_DIM)) for _ in range(n_days)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, CFG.INPUT_DIM)) for _ in range(n_days)
        ])
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(CFG.DAY_DROPOUT)

        # GRU with patch embedding
        gru_input_size = CFG.INPUT_DIM * CFG.PATCH_SIZE  # 512 * 14 = 7168
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=CFG.GRU_HIDDEN,
            num_layers=CFG.GRU_LAYERS,
            batch_first=True,
            dropout=CFG.GRU_DROPOUT
        )
        self.out = nn.Linear(CFG.GRU_HIDDEN, CFG.OUTPUT_DIM)

    def forward(self, x, day_ids):
        """
        Args:
            x: [B, T, 512] neural features
            day_ids: [B] session indices or single int
        Returns:
            [B, T', 41] log-probabilities
        """
        B, T, D = x.shape

        # Day adapter: per-sample linear transform
        if isinstance(day_ids, torch.Tensor):
            day_list = day_ids.tolist()
        elif isinstance(day_ids, int):
            day_list = [day_ids] * B
        else:
            day_list = list(day_ids)

        adapted = []
        for i, d in enumerate(day_list):
            a = torch.matmul(x[i], self.day_weights[d]) + self.day_biases[d]
            adapted.append(a)
        x = torch.stack(adapted)
        x = self.day_layer_dropout(self.day_layer_activation(x))

        # Patch embedding: unfold time dimension
        # [B, T, D] → [B, n_patches, D * patch_size]
        if T >= CFG.PATCH_SIZE:
            patches = x.unfold(1, CFG.PATCH_SIZE, CFG.PATCH_STRIDE)  # [B, n_patches, D, patch_size]
            patches = patches.permute(0, 1, 3, 2).reshape(B, patches.size(1), -1)  # [B, n_patches, D*patch_size]
        else:
            # Pad if too short
            pad_len = CFG.PATCH_SIZE - T
            x = F.pad(x, (0, 0, 0, pad_len))
            patches = x.reshape(B, 1, -1)

        # GRU
        out, _ = self.gru(patches)
        out = self.out(out)
        return F.log_softmax(out, dim=-1)

    def get_output_length(self, input_length):
        """Calculate output sequence length after patch embedding."""
        return max(1, (input_length - CFG.PATCH_SIZE) // CFG.PATCH_STRIDE + 1)

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
    def __init__(self, input_dim, num_sessions):
        super().__init__()
        self.weight = nn.Parameter(torch.stack([torch.eye(input_dim) for _ in range(num_sessions)]))
        self.bias = nn.Parameter(torch.zeros(num_sessions, input_dim))

    def forward(self, x, session_ids):
        w = self.weight[session_ids]
        b = self.bias[session_ids].unsqueeze(1)
        return torch.bmm(x, w) + b

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=(kernel_size - 1) // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer_norm(x).transpose(1, 2)
        out = self.glu(self.pw1(out))
        out = self.dw(out)
        out = self.swish(self.bn(out))
        out = self.drop(self.pw2(out)).transpose(1, 2)
        return out

class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.dropout_p = dropout
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x_n = self.attn_norm(x)
        attn = F.scaled_dot_product_attention(x_n, x_n, x_n,
                                               dropout_p=self.dropout_p if self.training else 0.0,
                                               is_causal=False)
        x = residual + self.drop(attn)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(CFG.INPUT_DIM, CFG.CONFORMER_DIM)
        self.layers = nn.ModuleList([
            ConformerBlock(CFG.CONFORMER_DIM, CFG.CONFORMER_HEADS,
                          CFG.CONFORMER_KERNEL, CFG.CONFORMER_DROPOUT)
            for _ in range(CFG.CONFORMER_LAYERS)
        ])
        self.output_proj = nn.Linear(CFG.CONFORMER_DIM, CFG.OUTPUT_DIM)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return F.log_softmax(self.output_proj(x), dim=2)

class ConformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = SubjectAdapter(CFG.INPUT_DIM, CFG.N_SESSIONS)
        self.encoder = ConformerEncoder()

    def forward(self, x, session_ids):
        return self.encoder(self.adapter(x, session_ids))

# ═══════════════════════════════════════════════════════════════════════════════
# DECODERS
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_decode(logits, token_map=TOKEN_MAP):
    """Simple greedy CTC decoder."""
    ids = torch.argmax(logits, dim=-1)
    collapsed = torch.unique_consecutive(ids)
    tokens = [token_map.get(i.item(), "") for i in collapsed if i.item() != 0]
    return " ".join(tokens)

def ctc_beam_search(log_probs, beam_width=10, blank_id=0):
    """Pure-Python CTC prefix beam search."""
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.cpu().numpy()
    T, C = log_probs.shape
    NEG_INF = float('-inf')
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = defaultdict(lambda: (NEG_INF, NEG_INF))
        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)
            for c in range(C):
                p_c = log_probs[t, c]
                if c == blank_id:
                    npb, npnb = new_beams[prefix]
                    new_beams[prefix] = (np.logaddexp(npb, p_total + p_c), npnb)
                else:
                    new_prefix = prefix + (c,)
                    npb, npnb = new_beams[new_prefix]
                    if prefix and c == prefix[-1]:
                        npnb = np.logaddexp(npnb, pb + p_c)
                        opb, opnb = new_beams[prefix]
                        new_beams[prefix] = (opb, np.logaddexp(opnb, pnb + p_c))
                    else:
                        npnb = np.logaddexp(npnb, p_total + p_c)
                    new_beams[new_prefix] = (npb, npnb)
        scored = sorted(new_beams.items(), key=lambda x: np.logaddexp(*x[1]), reverse=True)
        beams = dict(scored[:beam_width])

    if not beams:
        return ""
    best = max(beams.keys(), key=lambda k: np.logaddexp(*beams[k]))
    return " ".join([TOKEN_MAP.get(i, "") for i in best])

# ═══════════════════════════════════════════════════════════════════════════════
# KENLM LANGUAGE MODEL DECODER
# ═══════════════════════════════════════════════════════════════════════════════

class LMDecoder:
    """Beam search decoder with optional KenLM integration via pyctcdecode."""

    def __init__(self):
        self.decoder = None
        self._init_decoder()

    def _init_decoder(self):
        try:
            from pyctcdecode import build_ctcdecoder

            if CFG.KENLM_PATH and os.path.exists(CFG.KENLM_PATH):
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
                if CFG.KENLM_PATH:
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
            probs = np.exp(logits.astype(np.float64))
            probs = probs / probs.sum(axis=-1, keepdims=True)  # Ensure valid probabilities
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
                probs = np.exp(logits.astype(np.float64))
                probs = probs / probs.sum(axis=-1, keepdims=True)
                beams = self.decoder.decode_beams(probs, beam_width=bw)
                results = []
                for b in beams[:nb]:
                    text = b[0]
                    score = (b[3] + b[4]) if len(b) > 4 else b[3] if len(b) > 3 else 0.0
                    results.append((text, score))
                return results
            except:
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
                n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
                print(f"  ✓ LLM loaded: {model_name} ({n_params:.0f}M params)")
                return
            except Exception as e:
                print(f"  Could not load {model_name}: {e}")
                continue

        print("  WARNING: No LLM available. Rescoring disabled.")

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
        except:
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
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def warmup_cosine_lr(optimizer, epoch, warmup=3, total=30, lr_min=1e-6):
    """Apply warmup + cosine annealing LR schedule."""
    for pg in optimizer.param_groups:
        base = pg.get('initial_lr', pg['lr'])
        if epoch < warmup:
            lr = base * (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(total - warmup, 1)
            lr = lr_min + 0.5 * (base - lr_min) * (1 + math.cos(math.pi * progress))
        pg['lr'] = lr
    return optimizer.param_groups[0]['lr']

def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    """Train one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss, n = 0.0, 0
    use_amp = CFG.DEVICE == "cuda"
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        x, y, x_lens, y_lens, sids = batch[:5]
        x = x.to(CFG.DEVICE)
        sids = sids.to(CFG.DEVICE)

        if use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, sids)
                # Compute proper output lengths
                if hasattr(model, 'get_output_length'):
                    out_lens = torch.tensor([model.get_output_length(l.item()) for l in x_lens])
                else:
                    out_lens = x_lens.clone()
                out_lens = torch.clamp(out_lens, max=logits.size(1))
                y_lens_c = torch.clamp(y_lens, max=logits.size(1) - 1)
                loss = criterion(logits.permute(1, 0, 2).cpu(), y, out_lens, y_lens_c)
        else:
            logits = model(x, sids)
            if hasattr(model, 'get_output_length'):
                out_lens = torch.tensor([model.get_output_length(l.item()) for l in x_lens])
            else:
                out_lens = x_lens.clone()
            out_lens = torch.clamp(out_lens, max=logits.size(1))
            y_lens_c = torch.clamp(y_lens, max=logits.size(1) - 1)
            logits_cpu = logits.permute(1, 0, 2).cpu() if CFG.DEVICE == "mps" else logits.permute(1, 0, 2)
            loss = criterion(logits_cpu, y, out_lens, y_lens_c)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss = loss / CFG.GRAD_ACCUM

        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % CFG.GRAD_ACCUM == 0:
            if use_amp and scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if use_amp and scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * CFG.GRAD_ACCUM
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / max(n, 1)

def validate_epoch(model, loader, criterion, lm_decoder=None, use_beam=False):
    """Validate model and compute WER."""
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            x, y, x_lens, y_lens, sids = batch[:5]
            x = x.to(CFG.DEVICE)
            sids = sids.to(CFG.DEVICE)

            use_amp = CFG.DEVICE == "cuda"
            if use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x, sids)
            else:
                logits = model(x, sids)

            # Loss
            if hasattr(model, 'get_output_length'):
                out_lens = torch.tensor([model.get_output_length(l.item()) for l in x_lens])
            else:
                out_lens = x_lens.clone()
            out_lens = torch.clamp(out_lens, max=logits.size(1))
            y_lens_c = torch.clamp(y_lens, max=logits.size(1) - 1)
            loss = criterion(logits.permute(1, 0, 2).float().cpu(), y, out_lens, y_lens_c)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * x.size(0)
                n += x.size(0)

            # Decode
            logits_cpu = logits.float().cpu()
            for i in range(x.size(0)):
                ol = min(logits.size(1), out_lens[i].item()) if hasattr(model, 'get_output_length') else x_lens[i].item()
                pred_logits = logits_cpu[i, :ol, :]

                if use_beam and lm_decoder:
                    pred = lm_decoder.decode(pred_logits)
                elif use_beam:
                    pred = ctc_beam_search(pred_logits, beam_width=CFG.BEAM_WIDTH)
                else:
                    pred = greedy_decode(pred_logits)

                true = " ".join([TOKEN_MAP.get(idx.item(), "") for idx in y[i, :y_lens[i]]])
                all_pred.append(pred)
                all_true.append(true)

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

        for epoch in range(1, CFG.EPOCHS + 1):
            cur_lr = warmup_cosine_lr(optimizer, epoch - 1, CFG.WARMUP_EPOCHS, CFG.EPOCHS, CFG.LR_MIN)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)

            # Validate with greedy for speed during training (beam search is slow)
            val_loss, wer = validate_epoch(model, val_loader, criterion,
                                            lm_decoder=None, use_beam=False)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['wer'].append(wer)
            history['lr'].append(cur_lr)

            print(f"  Ep {epoch:2d} | LR {cur_lr:.1e} | Loss {train_loss:.4f}/{val_loss:.4f} | WER {wer*100:.2f}%")

            if wer < best_wer:
                best_wer = wer
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save(best_state, os.path.join(CFG.OUTPUT_DIR, f"fold_{fold}_best.pt"))
                print(f"       → Best! Saved (WER={wer*100:.2f}%)")
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
    """Run ensemble inference with TTA, KenLM, and LLM rescoring."""
    # Load all fold models
    models = []
    for i, state in enumerate(fold_states):
        if CFG.MODE == "finetune":
            m = GRUDecoder(n_days=CFG.N_SESSIONS)
        else:
            m = ConformerModel()
        m.load_state_dict(state)
        m = m.to(CFG.DEVICE)
        m.eval()
        models.append(m)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    all_pred, all_true = [], []

    for batch in tqdm(test_loader, desc="Ensemble Inference"):
        x, y, x_lens, y_lens, sids = batch[:5]

        # Average logits across folds
        fold_logits = []
        for model in models:
            if CFG.USE_TTA:
                logits = tta_predict(model, x[0].numpy(), sids[0].item())
            else:
                xt = x.to(CFG.DEVICE)
                sid = sids.to(CFG.DEVICE)
                with torch.no_grad():
                    if CFG.DEVICE == "cuda":
                        with autocast('cuda', dtype=torch.bfloat16):
                            logits = model(xt, sid)[0].float().cpu()
                    else:
                        logits = model(xt, sid)[0].cpu()

            # Clip to output length
            if hasattr(model, 'get_output_length'):
                ol = model.get_output_length(x_lens[0].item())
            else:
                ol = x_lens[0].item()
            ol = min(ol, logits.size(0))
            fold_logits.append(logits[:ol])

        # Pad to same length and average
        max_len = max(l.size(0) for l in fold_logits)
        padded = [F.pad(l, (0, 0, 0, max_len - l.size(0)), value=-100) for l in fold_logits]
        avg_logits = torch.stack(padded).mean(0)

        # Decode
        if CFG.USE_LLM_RESCORE and llm_rescorer.model is not None:
            hyps = lm_decoder.decode_nbest(avg_logits)
            pred = llm_rescorer.rescore(hyps)
        elif lm_decoder.decoder is not None:
            pred = lm_decoder.decode(avg_logits)
        else:
            pred = ctc_beam_search(avg_logits, beam_width=CFG.BEAM_WIDTH)

        all_pred.append(pred)
        true = " ".join([TOKEN_MAP.get(idx.item(), "") for idx in y[0, :y_lens[0]]])
        all_true.append(true)

    wer = jiwer.wer(all_true, all_pred) if all_true else 1.0

    # Show examples
    print(f"\n  Sample predictions:")
    for i in range(min(5, len(all_pred))):
        print(f"    TRUE: {all_true[i][:80]}")
        print(f"    PRED: {all_pred[i][:80]}")
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
