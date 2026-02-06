import pandas as pd
import numpy as np
import logging
import sys
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

n_splits = 5
seed = 42

resfile = 'results/tfidf-logit-singleclass_folds'+str(n_splits)+'_seed'+str(seed)
logfile = resfile+'.txt'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(logfile, mode='w')
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


df = pd.read_csv('../../Data/Corpuses/evurbth_core_KD-ANNOTATED.csv',header=None)
df = df.iloc[:,[0,3]]
df.columns = ['text','label']
df = df.dropna()
df = df[df['label'].isin(['model','method','theory','empirical'])]
X = df['text']
y = df['label']

logger.info("Class distribution after filtering:")
logger.info(df['label'].value_counts())

pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

metrics = {
    "accuracy": [],
    "f1_macro": [],
    "f1_weighted": []
}
zeror_metrics = {"accuracy": [], "f1_macro": [],"f1_weighted": []}

kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)


for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    zeror = DummyClassifier(strategy="most_frequent")
    zeror.fit(X_train, y_train)
    y_zeror = zeror.predict(X_test)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    metrics["accuracy"].append(acc)
    metrics["f1_macro"].append(f1_macro)
    metrics["f1_weighted"].append(f1_weighted)

    zeror_metrics["accuracy"].append(accuracy_score(y_test, y_zeror))
    zeror_metrics["f1_macro"].append(f1_score(y_test, y_zeror, average='macro'))
    zeror_metrics["f1_weighted"].append(f1_score(y_test, y_zeror, average='weighted'))

    logger.info(f"Fold {fold+1} | Acc: {acc:.3f} | Macro-F1: {f1_macro:.3f}")

    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    for i, class_label in enumerate(pipeline.named_steps['clf'].classes_):
        top_indices = np.argsort(pipeline.named_steps['clf'].coef_[i])[-5:]
        top_words = [feature_names[j] for j in top_indices]
        logger.info(f"Top words for {class_label}: {top_words}")




logger.info(f"\n\n\n--- Indicators ---")
logger.info(f"{'Metric':<15} | {'TF-IDF Mean (Std)':<20} | {'ZeroR Mean'}")
logger.info("-"*50)

for m, values in metrics.items():
    tf_mean, tf_std = np.mean(metrics[m]), np.std(metrics[m])
    zr_mean = np.mean(zeror_metrics[m])
    logger.info(f"{m:<15} | {tf_mean:.3f} (+/- {tf_std:.3f}) | {zr_mean:.3f}")
logger.info("="*50)


#  Confusion Matrix for the final fold
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(resfile+".png", bbox_inches='tight')
plt.close()

