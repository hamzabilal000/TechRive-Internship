# Week 2 — Deep Learning (ANN, CNN) & NLP

Neural networks and text. An ANN and a CNN both classify handwritten digits, and
a TF-IDF + Logistic Regression model classifies short support messages.

> **Environment note:** PyTorch and TensorFlow could not be installed in this
> environment, so — exactly as the task allows — the ANN uses scikit-learn's
> `MLPClassifier` and the CNN is implemented **from scratch in pure NumPy**
> (forward pass *and* backpropagation). This is arguably more instructive: nothing
> is hidden behind a framework.

## What was done

### 1. ANN — `01_ann_mlp_digits.py`
- A fully-connected network (`64 → 64 → 32 → 10`, ReLU, Adam) trained on the
  8×8 **digits** image dataset with `MLPClassifier`.
- Features standardised first so the optimiser converges cleanly.
- **Result: ~0.967 test accuracy.**

### 2. CNN from scratch — `02_cnn_from_scratch.py`
- A minimal CNN built entirely in NumPy:
  `Conv2D(8×3×3) → ReLU → MaxPool2×2 → Flatten → Dense(72→10) → Softmax`.
- Implements both the **forward pass and the backward pass** (conv gradients,
  max-pool gradient routing via an argmax mask, softmax+cross-entropy gradient),
  trained with mini-batch SGD.
- **Result: ~0.95 test accuracy** on the 8×8 digits, in ~4 seconds.

### 3. NLP — `03_nlp_tfidf.py`
- A self-contained labelled set of 57 short support messages in three categories
  (`billing`, `technical`, `general`).
- **TF-IDF** vectorisation (unigrams, English stop-words) + **Logistic Regression**.
- **Result: ~0.78 test accuracy**, and it correctly routes three unseen messages.
- Fully offline/deterministic — the dataset lives in the script, no download.

## Representative results

| Task | Approach | Test accuracy |
|------|----------|---------------|
| Digits (ANN) | sklearn MLPClassifier | ~0.967 |
| Digits (CNN) | from-scratch NumPy CNN | ~0.95 |
| Support triage (NLP) | TF-IDF + LogReg | ~0.78 |

## Tools used
`Python 3.11`, `numpy`, `scikit-learn`.

## Key learnings
- An MLP treats an image as a flat pixel vector; a **CNN** exploits spatial
  structure with shared-weight filters, which is why so few parameters still learn
  digit shapes well.
- Writing convolution/pooling backprop by hand makes the chain rule concrete:
  max-pool simply routes each gradient back to the position that "won", and the
  softmax + cross-entropy gradient collapses to the clean `probs − onehot`.
- Plain SGD on a tiny net is sensitive to learning rate — too high and the loss
  spikes on unlucky batches (this is exactly what momentum/Adam smooth out).
- **TF-IDF + Logistic Regression** is a remarkably strong, fast text baseline;
  on small data, unigrams generalise better than bigrams (fewer, denser features).
- This support-triage classifier is a deliberate lead-in to the capstone's
  `/triage` theme in later weeks.

## How to run
```bash
pip install -r requirements.txt
python 01_ann_mlp_digits.py
python 02_cnn_from_scratch.py
python 03_nlp_tfidf.py
```

---
AI assistance (Claude) was used for debugging, explaining concepts, and code review while completing this week's tasks.
