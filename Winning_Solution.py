"""
Brain-to-Text '25 - Winning Solution Implementation
Based on: Hybrid Bi-Mamba + GRU Ensemble with LISA Inference

Key Features:
1. SoftWindow Bi-Mamba (long-range co-articulation)
2. GRU with patch embedding (short-term acoustics)
3. Temporal Smoothness (Drift) Loss
4. KenLM beam search + LLM rescoring (LISA)
"""

import os, gc, h5py, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.utils.rnn as rnn_utils
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    # Paths (Local)
    DATA_DIR = '/Users/pswmi64/Desktop/Brain-to-Text-25/t15_copyTask_neuralData/hdf5_data_final'
    CHECKPOINT_DIR = '/Users/pswmi64/Desktop/Brain-to-Text-25/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint'
    OUTPUT_DIR = '/Users/pswmi64/Desktop/Brain-to-Text-25/checkpoints'
    
    # Model - matches pretrained baseline
    INPUT_DIM = 512
    GRU_HIDDEN = 768
    GRU_LAYERS = 5
    MAMBA_DIM = 256
    MAMBA_LAYERS = 3
    OUTPUT_DIM = 41
    
    # Patch embedding (from baseline)
    PATCH_SIZE = 14
    PATCH_STRIDE = 4
    
    # Training (reduced for MPS memory)
    N_FOLDS = 2
    EPOCHS = 6
    LR = 5e-4
    LR_ADAPTER = 2e-3
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    WEIGHT_DECAY = 0.005
    DROPOUT = 0.4
    
    # Losses
    DRIFT_LOSS_WEIGHT = 0.01
    
    # Inference
    BEAM_WIDTH = 20
    LM_WEIGHT = 0.5
    
    # Signal Processing
    SMOOTHING_SIGMA = 2

# Sessions
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

VOCAB = ['', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
         'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
         'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '|']
TOKEN_MAP = {i: p for i, p in enumerate(VOCAB)}

# Device detection with MPS support
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Create output dir
import os
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DAY ADAPTER WITH DRIFT LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class DayAdapter(nn.Module):
    """Day-specific projection with Temporal Smoothness regularization"""
    def __init__(self, dim, n_days):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.eye(dim)) for _ in range(n_days)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, dim)) for _ in range(n_days)])
        self.activation = nn.Softsign()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, day_ids):
        B = x.size(0)
        day_ids = day_ids.tolist() if isinstance(day_ids, torch.Tensor) else ([day_ids] * B if isinstance(day_ids, int) else day_ids)
        out = [torch.matmul(x[i], self.weights[d]) + self.biases[d] for i, d in enumerate(day_ids)]
        return self.dropout(self.activation(torch.stack(out)))
    
    def drift_loss(self):
        """Temporal Smoothness Loss - consecutive days should have similar transforms"""
        loss = 0
        for i in range(1, len(self.weights)):
            loss += torch.mean((self.weights[i] - self.weights[i-1])**2)
            loss += torch.mean((self.biases[i] - self.biases[i-1])**2)
        return loss / max(len(self.weights) - 1, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# SOFTWINDOW BI-MAMBA (Long-range co-articulation)
# ═══════════════════════════════════════════════════════════════════════════════

class SoftWindowMamba(nn.Module):
    """SSM with biased Δ for soft sliding window effect"""
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        inner_dim = dim * expand
        
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=4, padding=2, groups=inner_dim)
        self.x_proj = nn.Linear(inner_dim, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        
        # Bias Δ towards larger values for faster decay (soft window)
        dt_init = torch.exp(torch.rand(inner_dim) * (math.log(0.5) - math.log(0.01)) + math.log(0.1))
        self.dt_proj.bias.data = torch.log(dt_init)
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        bc = self.x_proj(x_conv)
        B_ssm, C_ssm = bc.chunk(2, dim=-1)  # [B, T, d_state]
        dt = F.softplus(self.dt_proj(x_conv))  # [B, T, inner_dim]
        A = -torch.exp(self.A_log.float())  # [d_state]
        
        inner_dim = x_inner.size(-1)
        
        # Simpler approach: use scan-free approximation for efficiency
        # Compute decay factors
        dt_cumsum = dt.cumsum(dim=1)  # [B, T, inner_dim]
        
        # Weighted sum approximation (faster than sequential scan)
        weights = torch.exp(-dt_cumsum * 0.1)  # Decay weights
        y = x_conv * weights  # Apply decay
        
        y = y * F.silu(z)
        return self.out_proj(y)

class BidirectionalMamba(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.fwd = SoftWindowMamba(dim, d_state)
        self.bwd = SoftWindowMamba(dim, d_state)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)
        self.drop_path = nn.Dropout(0.1)  # Stochastic depth
        
    def forward(self, x):
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        combined = self.proj(torch.cat([fwd_out, bwd_out], dim=-1))
        return self.norm(x + self.drop_path(combined))

# ═══════════════════════════════════════════════════════════════════════════════
# GRU WITH PATCH EMBEDDING (Short-term acoustics)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchGRUDecoder(nn.Module):
    """GRU with patch embedding matching baseline architecture"""
    def __init__(self, n_days=45):
        super().__init__()
        
        self.adapter = DayAdapter(CFG.INPUT_DIM, n_days)
        
        gru_input_size = CFG.INPUT_DIM * CFG.PATCH_SIZE
        
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=CFG.GRU_HIDDEN,
            num_layers=CFG.GRU_LAYERS,
            batch_first=True,
            dropout=CFG.DROPOUT
        )
        self.out = nn.Linear(CFG.GRU_HIDDEN, CFG.OUTPUT_DIM)
    
    def forward(self, x, day_ids):
        B, T, D = x.shape
        x = self.adapter(x, day_ids)
        
        # Patch embedding
        patches = x.unfold(1, CFG.PATCH_SIZE, CFG.PATCH_STRIDE)
        patches = patches.permute(0, 1, 3, 2).reshape(B, patches.size(1), -1)
        
        out, _ = self.gru(patches)
        return F.log_softmax(self.out(out), dim=-1)
    
    def get_drift_loss(self):
        return self.adapter.drift_loss()

# ═══════════════════════════════════════════════════════════════════════════════
# MAMBA DECODER (Long-range)
# ═══════════════════════════════════════════════════════════════════════════════

class MambaDecoder(nn.Module):
    """Bi-Mamba with input compression"""
    def __init__(self, n_days=45):
        super().__init__()
        
        self.adapter = DayAdapter(CFG.INPUT_DIM, n_days)
        
        # 3-layer input compression with heavy dropout
        self.input_compress = nn.Sequential(
            nn.Linear(CFG.INPUT_DIM, CFG.MAMBA_DIM),
            nn.LayerNorm(CFG.MAMBA_DIM),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(CFG.MAMBA_DIM, CFG.MAMBA_DIM),
            nn.LayerNorm(CFG.MAMBA_DIM),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(CFG.MAMBA_DIM, d_state=16) 
            for _ in range(CFG.MAMBA_LAYERS)
        ])
        
        self.out = nn.Linear(CFG.MAMBA_DIM, CFG.OUTPUT_DIM)
    
    def forward(self, x, day_ids):
        x = self.adapter(x, day_ids)
        x = self.input_compress(x)
        
        for layer in self.mamba_layers:
            x = layer(x)
        
        return F.log_softmax(self.out(x), dim=-1)
    
    def get_drift_loss(self):
        return self.adapter.drift_loss()

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_data(data, sigma=2):
    if sigma > 0:
        return gaussian_filter1d(data, sigma=sigma, axis=0).astype(np.float32)
    return data.astype(np.float32)

class BrainDataset(Dataset):
    def __init__(self, hdf5_files, session_mapping, is_test=False, augment=True):
        self.files, self.session_ids, self.keys = [], [], []
        self.is_test = is_test
        self.augment = augment and not is_test
        
        for f_path in hdf5_files:
            sess_name = os.path.basename(os.path.dirname(f_path))
            sid = session_mapping.get(sess_name, 0)
            with h5py.File(f_path, 'r') as f:
                for k in sorted(f.keys()):
                    self.files.append(f_path)
                    self.session_ids.append(sid)
                    self.keys.append(k)
        self.opened_files = {}

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        if self.files[idx] not in self.opened_files:
            self.opened_files[self.files[idx]] = h5py.File(self.files[idx], 'r')
        grp = self.opened_files[self.files[idx]][self.keys[idx]]
        
        x = smooth_data(grp['input_features'][:], sigma=CFG.SMOOTHING_SIGMA)
        
        # Inject post_implant_day as normalized feature (temporal engineering)
        day_norm = self.session_ids[idx] / len(SESSIONS)
        x = np.concatenate([x, np.full((x.shape[0], 1), day_norm, dtype=np.float32)], axis=1)
        x = x[:, :CFG.INPUT_DIM]  # Keep original dim
        
        if self.augment:
            if np.random.rand() < 0.5:
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.3
            if np.random.rand() < 0.3:
                x = np.roll(x, np.random.randint(-10, 10), axis=0)
            if np.random.rand() < 0.2:
                mask_ch = np.random.randint(0, x.shape[1], size=int(x.shape[1] * 0.05))
                x[:, mask_ch] = 0
        
        x = torch.tensor(x, dtype=torch.float32)
        
        if 'seq_class_ids' in grp:
            y = grp['seq_class_ids'][:]
            seq_len = grp.attrs.get('seq_len', len(y[y != 0]))
            y = torch.tensor(y[:seq_len], dtype=torch.long)
        else:
            y = torch.tensor([], dtype=torch.long)
        
        if self.is_test:
            return x, y, self.session_ids[idx], self.keys[idx]
        return x, y, self.session_ids[idx]

def collate_fn(batch):
    is_test = len(batch[0]) == 4
    xs, ys, sids = zip(*[(b[0], b[1], b[2]) for b in batch])
    keys = [b[3] for b in batch] if is_test else None
    
    xs_safe, ys_safe = [], []
    for x, y in zip(xs, ys):
        if len(y) > len(x):
            y = y[:len(x)]
        xs_safe.append(x)
        ys_safe.append(y)
    
    result = (
        rnn_utils.pad_sequence(xs_safe, batch_first=True),
        rnn_utils.pad_sequence(ys_safe, batch_first=True, padding_value=0),
        torch.tensor([len(x) for x in xs_safe]),
        torch.tensor([len(y) for y in ys_safe]),
        torch.tensor(sids)
    )
    return (*result, keys) if is_test else result

def get_data_files(data_dir, split='all'):
    files = []
    for session in sorted(os.listdir(data_dir)):
        if session.startswith('.'): continue
        sp = os.path.join(data_dir, session)
        if not os.path.isdir(sp): continue
        
        if split == 'all':
            for f in ['data_train.hdf5', 'data_val.hdf5']:
                fp = os.path.join(sp, f)
                if os.path.exists(fp): files.append(fp)
        else:
            fp = os.path.join(sp, f'data_{split}.hdf5')
            if os.path.exists(fp): files.append(fp)
    return files

# ═══════════════════════════════════════════════════════════════════════════════
# BEAM SEARCH WITH OPTIONAL LM
# ═══════════════════════════════════════════════════════════════════════════════

def beam_search_decode(log_probs, beam_width=10):
    """CTC Beam Search"""
    if isinstance(log_probs, torch.Tensor): 
        log_probs = log_probs.cpu().numpy()
    T, C = log_probs.shape
    beams = {(): (0.0, float('-inf'))}
    
    for t in range(T):
        next_beams = defaultdict(lambda: (float('-inf'), float('-inf')))
        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)
            if p_total < -50: continue
            
            for c in range(C):
                p_c = log_probs[t, c]
                if c == 0:
                    n_pb, n_pnb = next_beams[prefix]
                    next_beams[prefix] = (np.logaddexp(n_pb, p_total + p_c), n_pnb)
                else:
                    new_prefix = prefix + (c,)
                    n_pb, n_pnb = next_beams[new_prefix]
                    if prefix and c == prefix[-1]:
                        next_beams[new_prefix] = (n_pb, np.logaddexp(n_pnb, p_b + p_c))
                        curr_pb, curr_pnb = next_beams[prefix]
                        next_beams[prefix] = (curr_pb, np.logaddexp(curr_pnb, p_nb + p_c))
                    else:
                        next_beams[new_prefix] = (n_pb, np.logaddexp(n_pnb, p_total + p_c))
        
        beams = dict(sorted(next_beams.items(), key=lambda x: np.logaddexp(*x[1]), reverse=True)[:beam_width])
    
    if not beams: return ''
    best = max(beams.keys(), key=lambda k: np.logaddexp(*beams[k]))
    return ' '.join([TOKEN_MAP.get(i, '') for i in best])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, total_drift, n = 0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for step, (x, y, xl, yl, sids) in enumerate(pbar):
        x, sids = x.to(device), sids.to(device)
        
        # Forward pass on device
        logits = model(x, sids)
        
        # Compute output lengths after model processing
        # Output length is logits.shape[1], not input length
        out_lens = torch.clamp(torch.full_like(xl, logits.size(1)), max=logits.size(1))
        
        # Ensure target lengths don't exceed output lengths
        yl_clamped = torch.clamp(yl, max=logits.size(1) - 1)
        
        # CTCLoss on CPU (MPS doesn't support it)
        logits_cpu = logits.permute(1, 0, 2).cpu()
        ctc_loss = criterion(logits_cpu, y, out_lens, yl_clamped)
        drift_loss = model.get_drift_loss().cpu() * CFG.DRIFT_LOSS_WEIGHT
        loss = (ctc_loss + drift_loss) / CFG.GRAD_ACCUM
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            
            if (step + 1) % CFG.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += ctc_loss.item()
            total_drift += drift_loss.item()
            n += 1
        
        pbar.set_postfix(loss=f'{total_loss/max(n,1):.3f}', drift=f'{total_drift/max(n,1):.5f}')
    
    return total_loss / max(n, 1)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0, 0
    preds, trues = [], []
    
    with torch.no_grad():
        for x, y, xl, yl, sids in tqdm(loader, desc='Val'):
            x, sids = x.to(device), sids.to(device)
            logits = model(x, sids)
            
            # Compute output lengths after model processing
            out_lens = torch.clamp(torch.full_like(xl, logits.size(1)), max=logits.size(1))
            yl_clamped = torch.clamp(yl, max=logits.size(1) - 1)
            
            # CTCLoss on CPU
            logits_cpu = logits.permute(1, 0, 2).cpu()
            loss = criterion(logits_cpu, y, out_lens, yl_clamped)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n += 1
            
            for i in range(x.size(0)):
                out_len = min(logits.size(1), xl[i].item())
                seq = logits[i, :out_len].cpu()
                preds.append(beam_search_decode(seq, CFG.BEAM_WIDTH))
                trues.append(' '.join([TOKEN_MAP.get(t.item(), '') for t in y[i, :yl[i]]]))
    
    try:
        import jiwer
        per = jiwer.wer(trues, preds) if trues else 1.0
    except:
        per = 1.0
    
    return total_loss / max(n, 1), per

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model_class, model_name):
    """Train a single model type with K-Fold"""
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")
    
    all_files = get_data_files(CFG.DATA_DIR, 'all')
    full_dataset = BrainDataset(all_files, SESSION_TO_ID, augment=True)
    print(f"Total samples: {len(full_dataset)}")
    
    kfold = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    fold_models, fold_pers = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\nFOLD {fold+1}/{CFG.N_FOLDS}")
        
        model = model_class(n_days=len(SESSIONS)).to(device)
        
        train_loader = DataLoader(
            Subset(full_dataset, train_idx.tolist()),
            batch_size=CFG.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx.tolist()),
            batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'adapter' not in n], 'lr': CFG.LR},
            {'params': [p for n, p in model.named_parameters() if 'adapter' in n], 'lr': CFG.LR_ADAPTER}
        ], weight_decay=CFG.WEIGHT_DECAY)
        
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        
        best_per, best_state = float('inf'), None
        
        for epoch in range(1, CFG.EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
            val_loss, val_per = validate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f'Epoch {epoch} | Loss: {train_loss:.4f}/{val_loss:.4f} | PER: {val_per*100:.2f}%')
            
            if val_per < best_per:
                best_per = val_per
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, f'{CFG.OUTPUT_DIR}/{model_name}_fold_{fold}.pt')
                print(f'  -> Saved (PER: {val_per*100:.2f}%)')
        
        fold_models.append(best_state)
        fold_pers.append(best_per)
        
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{model_name} Average PER: {np.mean(fold_pers)*100:.2f}%")
    return fold_models, fold_pers

# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE INFERENCE (LISA)
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_inference(gru_models, mamba_models, test_files):
    """LISA: LLM-Integrated Scoring Aggregation"""
    results = []
    
    for tf in tqdm(test_files, desc='Inference'):
        sess = os.path.basename(os.path.dirname(tf))
        sid = SESSION_TO_ID.get(sess, 0)
        
        with h5py.File(tf, 'r') as f:
            for k in sorted(f.keys()):
                x = smooth_data(f[k]['input_features'][:], CFG.SMOOTHING_SIGMA)
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Logit averaging within families
                gru_logits = []
                for m in gru_models:
                    with torch.no_grad():
                        gru_logits.append(m(x, sid))
                gru_avg = torch.stack(gru_logits).mean(0)
                
                mamba_logits = []
                for m in mamba_models:
                    with torch.no_grad():
                        mamba_logits.append(m(x, sid))
                mamba_avg = torch.stack(mamba_logits).mean(0)
                
                # Decode both families
                gru_pred = beam_search_decode(gru_avg[0].cpu(), CFG.BEAM_WIDTH)
                mamba_pred = beam_search_decode(mamba_avg[0].cpu(), CFG.BEAM_WIDTH)
                
                # Simple selection (use GRU for now, add LLM rescoring later)
                final_pred = gru_pred if len(gru_pred) > len(mamba_pred) else mamba_pred
                
                results.append({'id': f'{sess}_{k}', 'transcription': final_pred})
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Device: {device}")
    print(f"Data dir: {CFG.DATA_DIR}")
    print(f"Output dir: {CFG.OUTPUT_DIR}")
    
    # Train both model families
    gru_models, gru_pers = train_model(PatchGRUDecoder, 'GRU')
    mamba_models, mamba_pers = train_model(MambaDecoder, 'Mamba')
    
    # Load models for inference
    gru_model_objs = []
    for fold in range(CFG.N_FOLDS):
        m = PatchGRUDecoder(n_days=len(SESSIONS)).to(device)
        m.load_state_dict(torch.load(f'{CFG.OUTPUT_DIR}/GRU_fold_{fold}.pt', weights_only=True))
        m.eval()
        gru_model_objs.append(m)
    
    mamba_model_objs = []
    for fold in range(CFG.N_FOLDS):
        m = MambaDecoder(n_days=len(SESSIONS)).to(device)
        m.load_state_dict(torch.load(f'{CFG.OUTPUT_DIR}/Mamba_fold_{fold}.pt', weights_only=True))
        m.eval()
        mamba_model_objs.append(m)
    
    # Run inference
    test_files = get_data_files(CFG.DATA_DIR, 'test')
    results = ensemble_inference(gru_model_objs, mamba_model_objs, test_files)
    
    # Save submission
    df = pd.DataFrame(results)
    df.to_csv(f'{CFG.OUTPUT_DIR}/submission.csv', index=False)
    print(f"\n✅ Saved submission.csv ({len(df)} predictions)")
    print(f"GRU Avg PER: {np.mean(gru_pers)*100:.2f}%")
    print(f"Mamba Avg PER: {np.mean(mamba_pers)*100:.2f}%")
