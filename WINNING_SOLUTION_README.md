# Brain-to-Text '25 Winning Solution

## Quick Start

### For Kaggle:
1. Create new notebook
2. Copy contents of `Winning_Solution.py`
3. Add competition dataset: `brain-to-text-25`
4. Enable **GPU T4 x2**
5. Run all cells

### Key Components:

| Component | Purpose |
|-----------|---------|
| `SoftWindowMamba` | Long-range co-articulation (Δ biased for soft window) |
| `PatchGRUDecoder` | Short-term acoustics (patch_size=14, stride=4) |
| `DayAdapter` | Day-specific transforms with Temporal Smoothness Loss |
| `beam_search_decode` | CTC beam search (width=20) |

### Training Configuration:
```python
EPOCHS = 8
BATCH_SIZE = 8
GRAD_ACCUM = 4  # Effective batch = 32
N_FOLDS = 3
DRIFT_LOSS_WEIGHT = 0.01
```

### Important Losses:
1. **CTC Loss**: Standard sequence-to-sequence
2. **Drift Loss**: `L2(W[i] - W[i-1])` for adjacent day matrices

### To Add KenLM (for better WER):
```bash
pip install pyctcdecode kenlm
```

Then replace `beam_search_decode` with:
```python
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=VOCAB,
    kenlm_model_path="path/to/4gram.arpa",
    alpha=0.5,  # LM weight
    beta=1.0    # Word insertion bonus
)

def decode_with_lm(logits):
    return decoder.decode(logits.cpu().numpy())
```

## Expected Results

| Stage | PER/WER |
|-------|---------|
| Acoustic only (greedy) | ~15-20% |
| + Beam search | ~12-15% |
| + KenLM | ~5-8% |
| + LLM Rescoring (LISA) | ~1-3% |
