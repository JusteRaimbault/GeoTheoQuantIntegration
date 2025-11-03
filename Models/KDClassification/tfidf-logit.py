import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report
import re
import os

n_splits = 5
seed = 66

label_cols = ['Empirical','Theoretical','Modelling','Methodological']

df = pd.read_csv('corpus_ANNOTATED_TRANSLATED.csv')

for col in label_cols:
    print(col)
    df[col] = df[col].astype(int)

df['text'] = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')

X = df['text']
y = df[label_cols].values

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ('classifier', MultiOutputClassifier(
        LogisticRegression(
            solver='liblinear',
            random_state=seed,
            C=1.0
        )
    ))
])

kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

f1_micro_scores = []
f1_macro_scores = []
accuracy_scores = []
hamming_scores = []

f1_micro_zero_r = []
f1_macro_zero_r = []
acc_zero_r = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Zero-R baseline
    y_train_tuples = [tuple(row) for row in y_train]
    pattern_counts = pd.Series(y_train_tuples).value_counts()
    most_frequent_pattern = np.array(pattern_counts.index[0])
    n_test_samples = y_test.shape[0]
    y_pred_zero_r = np.tile(most_frequent_pattern, (n_test_samples, 1))
    acc_zero_r.append(accuracy_score(y_test, y_pred_zero_r))
    f1_micro_zero_r.append(f1_score(y_test, y_pred_zero_r, average='micro', zero_division=0))
    f1_macro_zero_r.append(f1_score(y_test, y_pred_zero_r, average='macro', zero_division=0))


    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_micro_scores.append(f1_micro)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro_scores.append(f1_macro)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    hamming = hamming_loss(y_test, y_pred)
    hamming_scores.append(hamming)

    #print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))

print(f"  Zero-R Accuracy: {np.mean(acc_zero_r):.4f} (+/- {np.std(acc_zero_r):.4f})")
print(f"  Zero-R Micro-F1 Score:     {np.mean(f1_micro_zero_r):.4f} (+/- {np.std(f1_micro_zero_r):.4f})")
print(f"  Zero-R Macro-F1 Score:     {np.mean(f1_macro_zero_r):.4f} (+/- {np.std(f1_macro_zero_r):.4f})")

print(f"  Average Exact Match Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
print(f"  Average Micro-F1 Score:     {np.mean(f1_micro_scores):.4f} (+/- {np.std(f1_micro_scores):.4f})")
print(f"  Average Macro-F1 Score:     {np.mean(f1_macro_scores):.4f} (+/- {np.std(f1_macro_scores):.4f})")
print(f"  Average Hamming Loss:       {np.mean(hamming_scores):.4f} (+/- {np.std(hamming_scores):.4f})")
