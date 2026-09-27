# Email Spam Detector

A machine learning classifier that detects spam emails/SMS using TF-IDF vectorization and a Naive Bayes model.

## Overview

This project classifies text messages as spam or ham (not spam) using classical machine learning. Built to understand the full ML pipeline: data loading, feature extraction, training, and evaluation.

## Dataset

SMS Spam Collection — 5,572 labeled messages (4,825 ham, 747 spam).

## Approach

1. **Preprocessing:** Split data into 80% train / 20% test
2. **Feature extraction:** TF-IDF vectorization (English stop words removed, top 3,000 features)
3. **Model:** Multinomial Naive Bayes
4. **Evaluation:** Precision, recall, F1-score, and confusion matrix

## Results

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.98 | 0.99 |
| Recall | 1.00 | 0.89 |
| F1-score | 0.99 | 0.94 |

**Overall accuracy:** 98%

**Confusion matrix:**

    [[965   1]
     [ 16 133]]

Out of 966 real emails, only 1 was wrongly flagged as spam. Out of 149 spam messages, 133 were caught.

## Setup

    pip install -r requirements.txt
    python spam.py

## Files

- `spam.py` — main script
- `SMSSpamCollection` — dataset
- `requirements.txt` — dependencies
