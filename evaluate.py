from email.parser import BytesParser
from email import policy
import os
from collections import Counter
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from naive_bayes import NaiveBayesBernoulli, create_matrix
from bert.bert_classfifer import BertClassifier, TextDataset

labels = {}
with open('SPAMTrain.label', 'r') as f:
    for line in f.readlines():
        parts = line.split()
        labels[parts[1]] = parts[0]

email_path = "TRAINING/"
rows = []
for file_name in tqdm(os.listdir(email_path), desc="Loading emails"):
    try:
        with open(os.path.join(email_path, file_name), 'rb') as fp:
            msg = BytesParser(policy=policy.compat32).parse(fp)
        body = None
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                break
        text = body.decode('utf-8', errors='replace') if body else ""
        rows.append((file_name, text, labels[file_name]))
    except Exception as e:
        print(f"Skipping {file_name}: {e}")

raw_df = pd.DataFrame(rows, columns=['file_name', 'text', 'label'])

train_files, test_files = train_test_split(
    raw_df['file_name'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=raw_df['label'].tolist(),
)
train_set = set(train_files)
test_set  = set(test_files)

train_df = raw_df[raw_df['file_name'].isin(train_set)].reset_index(drop=True)
test_df  = raw_df[raw_df['file_name'].isin(test_set)].reset_index(drop=True)

print(f"Train: {len(train_df)}  Test: {len(test_df)}")

print("\n=== Naive Bayes ===")

# Build vocabulary from training emails only
def is_meaningful(w):
    return w.isalpha() and len(w) > 2

spam_cnt, reg_cnt = Counter(), Counter()
for _, row in train_df.iterrows():
    words = row['text'].split()
    if row['label'] == '1':
        spam_cnt.update(w for w in words if is_meaningful(w))
    else:
        reg_cnt.update(w for w in words if is_meaningful(w))

spam_total = sum(spam_cnt.values()) or 1
reg_total  = sum(reg_cnt.values()) or 1
all_words  = set(list(spam_cnt)[:1000]) | set(list(reg_cnt)[:1000])
diffs = {
    w: abs(spam_cnt.get(w, 0)/spam_total - reg_cnt.get(w, 0)/reg_total)
    for w in all_words
}
features = [w for w, _ in sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:800]]

# Build feature matrices restricted to the split files
nb_full = create_matrix(training_dir=email_path, features=features)
nb_train = nb_full[nb_full['file_name'].isin(train_set)].reset_index(drop=True)
nb_test  = nb_full[nb_full['file_name'].isin(test_set)].reset_index(drop=True)

feat_cols = features
X_train_nb = nb_train[feat_cols]
y_train_nb = nb_train['label']
X_test_nb  = nb_test[feat_cols]
y_test_nb  = nb_test['label']

nb = NaiveBayesBernoulli()
nb.fit(X_train_nb, y_train_nb)

nb_preds = [nb.predict(X_test_nb.iloc[i])[0] for i in tqdm(range(len(X_test_nb)), desc="NB predict")]
nb_acc = np.mean(np.array(nb_preds) == y_test_nb.astype(int).values)
print(f"Naive Bayes accuracy: {nb_acc:.4f}")

r3print("\n=== BERT ===")

bc = BertClassifier()

train_dataset = TextDataset(
    train_df['text'].tolist(),
    train_df['label'].astype(int).tolist(),
    bc.tokenizer,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

bc.fit(loader=train_loader, epochs=3)

bert_preds = []
batch_size = 32
test_texts  = test_df['text'].tolist()
test_labels = test_df['label'].astype(int).tolist()

for i in tqdm(range(0, len(test_texts), batch_size), desc="BERT predict"):
    batch = test_texts[i:i+batch_size]
    bert_preds.extend(bc.predict(batch))

bert_acc = np.mean(np.array(bert_preds) == np.array(test_labels))
print(f"BERT accuracy: {bert_acc:.4f}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n=== Results ===")
print(f"Naive Bayes : {nb_acc:.4f}")
print(f"BERT        : {bert_acc:.4f}")
