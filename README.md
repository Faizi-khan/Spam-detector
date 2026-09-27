# Email Spam Detector

A machine learning classifier that detects spam emails/SMS using TF-IDF vectorization and two models: Naive Bayes and Logistic Regression.

## Overview

This project classifies text messages as spam or ham (not spam) using classical machine learning. Built to understand the full ML pipeline: data loading, feature extraction, training, and evaluation — including a comparison between two different models.

## Dataset

SMS Spam Collection — 5,572 labeled messages (4,825 ham, 747 spam).

## Approach

1. **Preprocessing:** Split data into 80% train / 20% test
2. **Feature extraction:** TF-IDF vectorization (English stop words removed, top 3,000 features)
3. **Models:** Multinomial Naive Bayes and Logistic Regression
4. **Evaluation:** Precision, recall, F1-score, and confusion matrix for both models

## Results

### Naive Bayes

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.98 | 0.99 |
| Recall | 1.00 | 0.89 |
| F1-score | 0.99 | 0.94 |

Confusion matrix:

    [[965   1]
     [ 16 133]]

### Logistic Regression

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.98 | 1.00 |
| Recall | 1.00 | 0.84 |
| F1-score | 0.99 | 0.91 |

Confusion matrix:

    [[966   0]
     [ 24 125]]

## Model Comparison

| Model | Ham Precision | Spam Precision | Spam Recall | False Positives |
|---|---|---|---|---|
| Naive Bayes | 0.98 | 0.99 | 0.89 | 1 |
| Logistic Regression | 0.98 | 1.00 | 0.84 | 0 |

Logistic Regression achieves zero false positives — no real email is ever misclassified as spam — at the cost of missing slightly more spam (24 vs. 16 false negatives). Naive Bayes catches more spam overall but carries a small risk of flagging real email. The right choice depends on whether false positives or false negatives are more costly for the use case: for a personal inbox, avoiding false positives (Logistic Regression) is usually preferred.

## Setup

    pip install -r requirements.txt
    python spam.py

## Files

- `spam.py` — main script (data loading, training, evaluation for both models)
- `SMSSpamCollection` — dataset
- `requirements.txt` — dependencies

## Next steps

- Add features beyond TF-IDF (message length, punctuation frequency, link presence)
- Wrap as an API endpoint
