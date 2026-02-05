import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import logging, sys


n_splits = 5
seed = 42

logfile = 'results/tfidf-logit_folds'+str(n_splits)+'_seed'+str(seed)+'.txt'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(logfile, mode='w') # 'w' overwrites, 'a' appends
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

label_cols = ['Empirical','Theoretical','Modelling','Methodological']

df = pd.read_csv('corpus_ANNOTATED_TRANSLATED.csv')

for col in label_cols:
    logger.info(col)
    df[col] = df[col].astype(int)

df['text'] = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')

X = df['text']
y = df[label_cols].values

logger.info('Class counts : '+str(y.sum(axis=0)))

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

# for precision/recall plot
all_y_true = []
all_y_probs = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    logger.info(f"\n\n--- Processing Fold {fold + 1}/{n_splits} ---")

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

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Output interpretability
    classifier = pipeline.named_steps['classifier']
    vectorizer = pipeline.named_steps['tfidf']
    feature_names = vectorizer.get_feature_names_out()
    for i, label in enumerate(label_cols):
        coef = classifier.estimators_[i].coef_[0]
        top_indices = np.argsort(coef)[-10:][::-1]
        bottom_indices = np.argsort(coef)[:10]
        logger.info(f"\n--- Top 10 words for: {label} ---")
        logger.info([feature_names[j] for j in top_indices])
        logger.info(f"--- Bottom 10 words (Strongest 'NOT' predictors): ---")
        logger.info([feature_names[j] for j in bottom_indices])

    # Apply the model to test data
    #y_pred = pipeline.predict(X_test)

    # probas for optimal threshold
    y_proba_train_list = pipeline.predict_proba(X_train)
    y_proba_test_list = pipeline.predict_proba(X_test)
    y_proba_train = np.column_stack([p[:, 1] for p in y_proba_train_list])
    y_proba_test = np.column_stack([p[:, 1] for p in y_proba_test_list])
    # optimal threshold based on train data only - micro-average is expected to be more stable on small N
    p, r, thresholds = precision_recall_curve(y_train.ravel(), y_proba_train.ravel())
    f1 = (2 * p * r) / (p + r + 1e-10)
    best_threshold = thresholds[np.argmax(f1)]
    y_pred_optimal = (y_proba_test >= best_threshold).astype(int)

    # Compute scores
    f1_micro = f1_score(y_test, y_pred_optimal, average='micro')
    f1_micro_scores.append(f1_micro)
    f1_macro = f1_score(y_test, y_pred_optimal, average='macro', zero_division=0)
    f1_macro_scores.append(f1_macro)
    acc = accuracy_score(y_test, y_pred_optimal)
    accuracy_scores.append(acc)
    hamming = hamming_loss(y_test, y_pred_optimal)
    hamming_scores.append(hamming)

    # store for plot
    y_proba_list = pipeline.predict_proba(X_test)
    all_y_true.append(y_test)
    all_y_probs.append(np.column_stack([p[:, 1] for p in y_proba_list]))

    #print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))

# Indicators
logger.info(f"\n\n\n--- Indicators ---")
logger.info(f"  Zero-R Accuracy: {np.mean(acc_zero_r):.4f} (+/- {np.std(acc_zero_r):.4f})")
logger.info(f"  Zero-R Micro-F1 Score:     {np.mean(f1_micro_zero_r):.4f} (+/- {np.std(f1_micro_zero_r):.4f})")
logger.info(f"  Zero-R Macro-F1 Score:     {np.mean(f1_macro_zero_r):.4f} (+/- {np.std(f1_macro_zero_r):.4f})")

logger.info(f"  Average Exact Match Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
logger.info(f"  Average Micro-F1 Score:     {np.mean(f1_micro_scores):.4f} (+/- {np.std(f1_micro_scores):.4f})")
logger.info(f"  Average Macro-F1 Score:     {np.mean(f1_macro_scores):.4f} (+/- {np.std(f1_macro_scores):.4f})")
logger.info(f"  Average Hamming Loss:       {np.mean(hamming_scores):.4f} (+/- {np.std(hamming_scores):.4f})")





# Precision-recall plot
y_true_flat = np.concatenate(all_y_true, axis = 0).ravel()
y_probs_flat = np.concatenate(all_y_probs, axis = 0).ravel()
precision, recall, _ = precision_recall_curve(y_true_flat, y_probs_flat)
ap_score = average_precision_score(y_true_flat, y_probs_flat)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'Micro-average PR (AP = {ap_score:.2f})', color='teal', lw=2)
prevalence = np.mean(y_true_flat)
plt.axhline(y=prevalence, color='red', linestyle='--', label=f'Random Baseline ({prevalence:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-averaged Precision-Recall Curve')
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.savefig('results/tfidf-logit-pr-curve_folds'+str(n_splits)+'_seed'+str(seed)+'.png', dpi=300, bbox_inches='tight')
plt.show()





# Best threshold
precision, recall, thresholds = precision_recall_curve(y_true_flat, y_probs_flat)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
logger.info(f"\n\n\n--- Global Optimal Threshold Results ---")
logger.info(f"Optimal Threshold: {best_threshold:.4f}")
logger.info(f"Max Micro-F1 Score: {best_f1:.4f}")
logger.info(f"Prevalence (Baseline): {np.mean(y_true_flat):.4f}")




