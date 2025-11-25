---
trigger: always_on
---

# Brain-to-Text '25 Competition Rules & Guidelines

## Overview
**Goal**: Develop algorithms to decode speech directly from brain activity (neural time series) into text.
**Context**: Helping people with ALS or brainstem stroke who have lost the ability to move and speak.
**Previous Benchmark**: Brain-to-Text '24 reduced WER from 11.06% to 5.77%.

## Computational Problem
- **Input**: Variable-length time series of neural activity.
- **Output**: Text (natural language).
- **Alignment**: No ground-truth alignment provided between neural activity and text.
- **Baseline Performance**: 6.70% Word Error Rate (WER) on the held-out test set.

## Critical Rules & Constraints
1.  **Generalization**: 
    - You **CANNOT** manually tell your model which corpus to use at test-time.
    - The model must be able to handle different types of speech (e.g., "random word" sentences vs. standard sentences) without manual intervention.
    - **Allowed**: The model *can* automatically detect which corpus/style to use.
2.  **Eligibility**: Submissions that violate the generalization rule (i.e., hardcoding corpus selection based on test set knowledge) are **not eligible** for prize money.

## Evaluation Metric
- **Primary Metric**: Word Error Rate (WER).
- **Goal**: Minimize WER below the baseline of 6.70%.

## Potential Avenues for Improvement
The following approaches are suggested and allowed:
- **Data Augmentation**: e.g., temporal masking.
- **Model Architecture**: e.g., Transformers, End-to-End models.
- **Loss Functions**: e.g., Transducer loss.
- **Language Modeling**: 
    - Using different LMs for different corpora (must be selected automatically).
    - Neural LMs instead of n-gram models.
- **Ensembling**: Combining multiple models.
- **Test-Time Techniques**:
    - Test time augmentation.
    - Test time adaptation (continuous finetuning).
- **Tokenization**: Syllables or Byte Pair Encoding (BPE) instead of phonemes.
- **Transfer Learning**: From ASR datasets or Brain-to-Text '24 (T12) neural dataset.

## Baseline Approach
The provided baseline uses:
1.  **Neural Decoder**: Decodes neural activity into phoneme probabilities.
2.  **Language Model**: n-gram model + LLM rescoring to convert phonemes to text.
