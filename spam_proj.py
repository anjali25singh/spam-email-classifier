# ============================================================
# Spam vs Ham: Traditional ML (TF-IDF) vs DistilBERT (PyTorch)
# ============================================================

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. LOAD + EXPLORE DATA
import kagglehub, glob
path = kagglehub.dataset_download("ashfakyeafi/spam-email-classification")
df = pd.read_csv(glob.glob(path + "/*.csv")[0])
print(df.head())
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Cleaning: missing rows, wrong labels and remove duplicates 
df = df.dropna(subset=["Category", "Message"])
df = df[df["Category"].isin(["ham", "spam"])]
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

df["label"] = (df["Category"] == "spam").astype(int)   

# Class distribution (dataset imbalanced)
print("\nClass distribution:\n", df["Category"].value_counts())
sns.countplot(x="Category", data=df)
plt.title("Class Distribution")
plt.savefig("class_distribution.png", dpi=150)
plt.show()


# 2. TRAIN / VAL / TEST SPLIT

X_temp, X_test, y_temp, y_test = train_test_split(
    df["Message"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=SEED)  

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


# Common evaluation function (Part A and Part B same metrics)

def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }


# 3. PART A: TRADITIONAL ML
def clean_text(text):
    text = text.lower()                       
    text = re.sub(r"http\S+", " url ", text)  
    text = re.sub(r"\d+", " num ", text)      
    text = re.sub(r"[^a-z\s]", " ", text)     
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_clean = X_train.apply(clean_text)
test_clean = X_test.apply(clean_text)

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
Xtr_tfidf = tfidf.fit_transform(train_clean)   
Xte_tfidf = tfidf.transform(test_clean)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
}

results = {}       # model name -> metrics
conf_mats = {}     # model name -> confusion matrix
train_times = {}   # model name -> seconds

for name, model in models.items():
    start = time.time()
    model.fit(Xtr_tfidf, y_train)
    train_times[name] = time.time() - start

    preds = model.predict(Xte_tfidf)
    results[name] = evaluate(y_test, preds)
    conf_mats[name] = confusion_matrix(y_test, preds)
    print(f"\n{name}: {results[name]}")


# 4. PART B: DISTILBERT (PyTorch training loop, no Trainer API)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

# Hyperparameters 
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64          
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5             
WEIGHT_DECAY = 0.01   

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def make_loader(texts, labels, shuffle):
    enc = tokenizer(list(texts), truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(labels.values))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# Layer freezing: embeddings freeze
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY)


def run_eval(loader):
    """Loader par loss, predictions aur true labels return karta hai."""
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item()
            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


history = {"train_loss": [], "val_loss": [], "val_acc": []}
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for ids, mask, labels in train_loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()                                   # purane gradients clear
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()                                     # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient explode na ho
        optimizer.step()                                        # weights update
        running_loss += out.loss.item()

    val_loss, val_preds, val_true = run_eval(val_loader)
    history["train_loss"].append(running_loss / len(train_loader))
    history["val_loss"].append(val_loss)
    history["val_acc"].append(accuracy_score(val_true, val_preds))
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss {history['train_loss'][-1]:.4f} "
          f"| val loss {val_loss:.4f} | val acc {history['val_acc'][-1]:.4f}")

train_times["DistilBERT"] = time.time() - start

# Final test evaluation (same metrics as Part A)
_, test_preds, test_true = run_eval(test_loader)
results["DistilBERT"] = evaluate(test_true, test_preds)
conf_mats["DistilBERT"] = confusion_matrix(test_true, test_preds)
print("\nDistilBERT:", results["DistilBERT"])

# ------------------------------------------------------------
# 5. VISUALIZATIONS
# ------------------------------------------------------------
sns.set_style("whitegrid")

# (a) Metric comparison across models
res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
res_long = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(9, 5))
sns.barplot(x="Metric", y="Score", hue="Model", data=res_long)
plt.ylim(0.8, 1.01)
plt.title("Metric Comparison Across Models")
plt.savefig("metric_comparison.png", dpi=150)
plt.show()

# (b) Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, conf_mats.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# (c) Training behavior of transformer
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.lineplot(x=list(epochs_range), y=history["train_loss"], marker="o", label="Train Loss", ax=axes[0])
sns.lineplot(x=list(epochs_range), y=history["val_loss"], marker="o", label="Val Loss", ax=axes[0])
axes[0].set_title("DistilBERT Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
sns.lineplot(x=list(epochs_range), y=history["val_acc"], marker="o", color="green", ax=axes[1])
axes[1].set_title("DistilBERT Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig("training_behavior.png", dpi=150)
plt.show()

# (d) Computational cost
plt.figure(figsize=(6, 4))
sns.barplot(x=list(train_times.keys()), y=list(train_times.values()))
plt.yscale("log")   # DistilBERT bahut slow hai, isliye log scale
plt.ylabel("Training time (seconds, log scale)")
plt.title("Computational Cost")
plt.savefig("training_time.png", dpi=150)
plt.show()

print("\nFinal Results:\n", res_df.round(4))
print("\nTraining times (s):", {k: round(v, 2) for k, v in train_times.items()})