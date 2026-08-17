# Vietnamese Text Correction

A statistical machine learning and probabilistic NLP framework for automated spelling, diacritic, and typographic error correction in Vietnamese text.

---

## 📖 Problem Formulation

Text correction aims to automatically transform noisy, corrupted, or misspelled text sentences $X = (x_1, x_2, \dots, x_T)$ into their correct ground truth forms $W = (w_1, w_2, \dots, w_T)$.

### Performance Metric
Model performance is evaluated using the **Character Error Rate (CER)** based on Levenshtein edit distance:

$$\text{CER} = \frac{\text{LevenshteinDistance}(\text{Predicted\_Text},\, \text{Ground\_Truth})}{\text{Length}(\text{Ground\_Truth})}$$

---

## 📊 Dataset Structure

- **`train.csv`**: Contains paired columns:
  - `input`: The noisy/corrupted Vietnamese sentence.
  - `corrected_text`: The ground truth sentence.
- **`test.csv`**: Contains evaluation samples with columns `id` and `input`.

---

## ⚙️ Implemented Algorithms

### 1. Baseline UTF-8 Cleaner
- Filters out corrupted UTF-8 byte sequences using defensive byte encoding and decoding fallbacks (`.encode('utf-8', 'ignore').decode('utf-8')`).

---

### 2. Statistical Machine Learning (Word-Level MLE)

Learns a word-level translation map directly from training data observations.

#### Alignment & Estimation
1. Filters sentence pairs where input and ground truth have equal lengths ($|X| = |W|$).
2. Computes the Maximum Likelihood Estimation (MLE) for word replacement:

   $$P(w_{\text{correct}} \mid w_{\text{input}}) = \frac{\text{Count}(w_{\text{input}},\, w_{\text{correct}})}{\sum_{w'} \text{Count}(w_{\text{input}},\, w')}$$

#### Inference
For each input word $x_i$, selects the most frequent ground-truth replacement:

$$\hat{w}_i = \arg\max_{w} P(w \mid x_i)$$

If $x_i$ is an Out-Of-Vocabulary (OOV) token unobserved during training, the algorithm falls back to identity mapping ($\hat{w}_i = x_i$).

---

### 3. Noisy Channel Model with Bigram Viterbi Decoding

Applies Bayes' Theorem to decouple sentence generation probability (Language Model) from error probability (Channel Model):

$$\hat{W} = \arg\max_{W} P(W \mid X) = \arg\max_{W} P(X \mid W) \cdot P(W)$$

#### A. Language Model $P(W)$
Models correct Vietnamese word sequences using Unigram and Bigram statistics in Log-space with Laplace smoothing:

$$\log P(w_t \mid w_{t-1}) = \log \left( \frac{\text{Count}(w_{t-1}, w_t) + 1}{\text{Count}(w_{t-1}) + |V|} \right)$$

For sentence initial tokens ($t=0$):

$$\log P(w_1) = \log \left( \frac{\text{Count}(w_1) + 1}{N + |V|} \right)$$

#### B. Channel Error Model $P(X \mid W)$
Calculates the likelihood of observing error word $x_t$ given intended word $w_t$:

$$\log P(x_t \mid w_t) = \log \left( \frac{\text{Count}(w_t, x_t)}{\text{Count}(w_t)} \right)$$

If an error pair $(x_t, w_t)$ was not observed in training data, a Levenshtein distance penalty is applied:

$$\log P(x_t \mid w_t) = \log \left( 0.01^{\text{dist}(x_t, w_t)} \right)$$

#### C. Viterbi Dynamic Programming Decoding
Finds the global optimal word sequence $\hat{W}$ over sentence length $T$:

1. **Trellis State Update ($t > 0$)**:

   $$V[t][w] = \log P(x_t \mid w) + \max_{w'} \left( V[t-1][w'] + \log P(w \mid w') \right)$$

2. **Backtracking**: Traces back from $\arg\max_w V[T][w]$ to reconstruct the globally optimal sentence.

---

## 💻 Environment & Installation

### Requirements
- **Python Version**: `python >= 3.10`
- **Dependencies**:
  ```bash
  pip install pandas numpy tqdm Levenshtein sentencepiece transformers torch
  ```

### Directory Structure
```text
Viet Text Correction/
├── train.csv                      # Training dataset
├── test.csv                       # Test dataset
├── viet_text_correction.ipynb     # Primary Jupyter Notebook
└── README.md                      # Documentation
```

---

## 🚀 Execution

Open `viet_text_correction.ipynb` in Jupyter Notebook or Google Colab and run:
- `run_statistical_ml_correction()`: Executes Word-Level MLE training and generates `statistical_ml_submission.csv`.
- `run_pure_viterbi_noisy_channel()`: Executes Viterbi Noisy Channel decoding and generates `pure_viterbi_noisy_channel_submission.csv`.
