# Spam Classification: Traditional ML vs DistilBERT

A comparative study of a classical NLP pipeline (TF-IDF + Logistic Regression / Naive Bayes) and a transformer model (DistilBERT fine-tuned with a custom PyTorch training loop) for classifying emails/SMS as **Spam** or **Ham**.

## Dataset
[Spam Email Classification (Kaggle)](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)
Download `spam.csv` and place it in the project root.

## Approaches
**Part A: Traditional ML**
- Text cleaning (lowercase, URL/number normalization, punctuation removal)
- TF-IDF features (unigrams + bigrams, 5000 features)
- Classifiers: Logistic Regression, Multinomial Naive Bayes

**Part B: Transformer**
- `distilbert-base-uncased` from Hugging Face
- Custom PyTorch training loop (no Trainer API)
- Embedding layer frozen, AdamW (lr 2e-5, weight decay 0.01), batch size 16, 3 epochs

Both parts use the same stratified train/test split and the same metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

## Visualizations
- Metric comparison across models
- Confusion matrices
- DistilBERT training behavior (loss and validation accuracy)
- Training time comparison

## How to Run
```bash
pip install -r requirements.txt
python spam_project.py
```
A GPU (e.g., Google Colab T4) is recommended for the DistilBERT part.

## Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.9661 | 0.8298 | 0.9141 | 0.8699 |
| Naive Bayes | 0.9748 | 1.0000 | 0.7969 | 0.8870 |
| DistilBERT | - | - | - | - |

*(Fill in after running the code.)*

## Limitations
- Dataset is small and imbalanced (far more ham than spam)
- Messages are short, so results may not generalize to long emails
- DistilBERT is much slower to train than the classical models
-