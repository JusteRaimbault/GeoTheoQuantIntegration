import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging, sys, tempfile
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)

# --- 1. CONFIG & FILTERING ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
target_classes = ['model','method','theory','empirical']
filename='../../Data/Corpuses/evurbth_core_KD-ANNOTATED.csv'

# parameters
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
N_SPLITS = 5
SEED = 42

resfile = ('results/scibert-singleclass_epochs'+str(NUM_TRAIN_EPOCHS)+'_lrate'+
           str(LEARNING_RATE)+'_batch'+str(BATCH_SIZE)+'_seqlength'+str(MAX_SEQ_LENGTH)+
           '_folds'+str(N_SPLITS)+'_seed'+str(SEED))
logfile = resfile+'.txt'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(logfile, mode='w') # 'w' overwrites, 'a' appends
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

temp_run_dir = tempfile.mkdtemp()


df = pd.read_csv(filename,header=None)
df = df.iloc[:,[0,3]]
df.columns = ['text','label']
df = df.dropna()
df = df[df['label'].isin(target_classes)]
X = df['text']
y = df['label']

# Map labels to integers
label2id = {label: i for i, label in enumerate(target_classes)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
results = {"scibert_acc": [], "scibert_f1": [], "zeror_acc": []}

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    train_ds = Dataset.from_pandas(df.iloc[train_idx]).map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH), batched=True)
    test_ds = Dataset.from_pandas(df.iloc[test_idx]).map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(target_classes), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=temp_run_dir,
        metric_for_best_model="f1_macro",
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=SEED
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    preds_out = trainer.predict(test_ds)
    probs = nn.functional.softmax(torch.tensor(preds_out.predictions), dim=-1).numpy()
    y_pred = np.argmax(probs, axis=-1)
    y_true = df.iloc[test_idx]['labels'].values

    zr = DummyClassifier(strategy="most_frequent").fit(df.iloc[train_idx]['text'], y_true)  # Simplified fit

    results["scibert_acc"].append(accuracy_score(y_true, y_pred))
    results["scibert_f1"].append(f1_score(y_true, y_pred, average='macro'))
    results["zeror_acc"].append(accuracy_score(y_true, zr.predict(df.iloc[test_idx]['text'])))

logger.info(f"SciBERT Acc: {np.mean(results['scibert_acc']):.4f} (+/- {np.std(results['scibert_acc']):.4f})")
logger.info(f"ZeroR Acc:   {np.mean(results['zeror_acc']):.4f}")