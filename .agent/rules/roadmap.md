---
trigger: always_on
---

Phase 1: Signal Processing & Data Augmentation (Your "Edge")
Goal: Make the inputs cleaner and the model more robust to noise.

The baseline feeds raw 512-dim features directly into the model. Neural data is noisy and non-stationary (it changes over time).

Feature smoothing (Gaussian Kernel):

The Logic: Neural spiking rates are often estimated by smoothing discrete spikes. The provided features are likely binned spike counts or powers.

Action: Apply a 1D Gaussian smoothing filter (sigma=20-40ms) across the time dimension for each channel before creating the dataset. This helps the model see "trends" rather than jittery noise.

Robust Normalization:

The Logic: Brain signals drift. A neuron firing at "100" today might fire at "80" tomorrow for the same movement.

Action: Instead of standard scaling (mean/std), use RobustScaler (Median and IQR) to ignore outliers/artifacts. Normalize per session or even per block to account for drift.

SpecAugment (Crucial for preventing overfitting):

The Logic: You don't have enough training data. SpecAugment (originally for audio) works perfectly for neural features.

Action: In your BrainDataset class, implement:

Time Masking: Set a random block of time steps to zero.

Channel Masking: Set a random block of the 512 feature channels to zero. This forces the model not to rely on any single specific neuron.

Phase 2: Architecture Upgrade (The "Moderate DL" Task)
Goal: Replace the simple RNN/Transformer with the current SOTA for signal-to-sequence tasks.

The baseline TransformerEncModel is okay, but it lacks the ability to capture local patterns effectively. The current standard for this task is the Conformer.

Implement a Conformer Encoder:

Why: A Conformer combines CNNs (great for local feature extraction, like "this spike happened right after that one") with Transformers (great for global context, like "this phoneme usually appears at the end of a sentence").

Structure:

Input -> Feed Forward -> Multi-Head Self Attention -> Convolution Module -> Feed Forward -> Layer Norm.

How: You don't need to write this from scratch. Use torchaudio.models.Conformer or the implementation from the speechbrain library.

Add a "Subject-Specific" Projection:

Why: If the competition has data from multiple days or sessions, the electrode alignment shifts slightly.

Action: Keep the adapter_layer from the baseline but make it trainable per-day (or per-session) if you have session IDs. This "aligns" different days into a common space before the deep model sees it.

Phase 3: The Decoding Strategy (The "Score Multiplier")
Goal: The baseline uses greedy_decoder (just picking the highest probability phoneme). This is where you will lose or win the competition.

You need to move from "Greedy" to "Beam Search" to "LLM Rescoring".

Implement CTC Beam Search:

The Logic: Greedy decoding fails if the model is 49% sure it's "A" and 51% sure it's "B". Beam search keeps the top-K possibilities (e.g., top 100) alive at every step.

Action: Use the pyctcdecode library. It is highly optimized and integrates easily with KenLM.

Integrate a 4-gram Language Model (KenLM):

The Logic: The brain data is noisy. If the acoustic model predicts "H-E-L-L-P", an n-gram model knows "HELLO" is more likely than "HELLP".

Action: Train a 4-gram KenLM on the provided competition corpus (or a large open text corpus like WikiText). Feed this into pyctcdecode to guide the beam search.

LLM Rescoring (Advanced):

The Logic: Beam search gives you the top 100 candidate sentences. A 4-gram model is stupid (it only sees 4 words). A large LLM (like Llama-3-8B or GPT-2) is smart.

Action:

Take the Top-N (e.g., 10) sentences from your beam search.

Feed them into a pre-trained LLM to get the "perplexity" (likelihood) of that sentence.

Re-rank the list based on the LLM's score. This corrects semantic errors like "I went to the store" vs "I went to the star".

Phase 4: Ensembling (The "Grandmaster" Touch)
Goal: Reduce variance.

k-Fold Cross-Validation:

The baseline just splits Train/Val/Test once. You should implement 5-Fold CV.

Train 5 separate Conformers on different splits of the data.

Logit Averaging:

For the final test set, run all 5 models.

Average their output probabilities (logits) before feeding them into the Beam Search decoder. This usually guarantees a 1-2% boost in accuracy.