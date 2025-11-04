# .venv/bin/pip install torch transformers datasets scikit-learn pandas accelerate -U
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import os

FILE_NAME = 'corpus_ANNOTATED_TRANSLATED.csv'
TEXT_COLUMN = 'text'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 256
RANDOM_STATE = 66
N_SPLITS = 5

df = pd.read_csv(FILE_NAME)
label_cols = ['Empirical','Theoretical','Modelling','Methodological']
NUM_LABELS = len(label_cols)
df['text'] = df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')

# Ensure labels are float32 for PyTorch loss function
df[label_cols] = df[label_cols].astype(np.float32)

# Create the Hugging Face Dataset object
hg_dataset = Dataset.from_pandas(df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(data):
    tokenized = tokenizer(data[TEXT_COLUMN],truncation=True,padding='max_length',max_length=MAX_SEQ_LENGTH)
    label_values = [data[col] for col in label_cols]
    transposed_labels = list(zip(*label_values))
    tokenized["labels"] = transposed_labels
    return tokenized

columns_to_remove = [TEXT_COLUMN] + label_cols

tokenized_hg_dataset = hg_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)


def compute_metrics(p):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    preds = (torch.sigmoid(torch.from_numpy(predictions)) > 0.5).numpy().astype(int)

    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    exact_match = accuracy_score(labels, preds)
    hamming = hamming_loss(labels, preds)

    return {'micro_f1': micro_f1,'macro_f1': macro_f1,'exact_match_acc': exact_match,'hamming_loss': hamming}


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
all_fold_metrics = []

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    print(f"\n" + "="*60)
    print(f"--- Fine-Tuning SciBERT: Fold {fold + 1}/{N_SPLITS} ---")

    train_dataset = tokenized_hg_dataset.select(train_index)
    test_dataset = tokenized_hg_dataset.select(test_index)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=f"./results_scibert_fold{fold+1}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        logging_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model='micro_f1',
        seed=RANDOM_STATE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("\nFold Evaluation Results:", eval_results)

    all_fold_metrics.append(eval_results)

print("\n" + "="*80)
print("SciBERT Cross-Validation Complete. Average Scores:")

results_df = pd.DataFrame(all_fold_metrics)
metric_cols = [col for col in results_df.columns if col.startswith('eval_')]

summary_data = {
    'Metric': [col.replace('eval_', '') for col in metric_cols],
    'SciBERT Model (Mean)': [results_df[col].mean() for col in metric_cols]
}

summary_df = pd.DataFrame(summary_data).round(4)
print(summary_df.to_markdown(index=False, numalign="left", stralign="left"))
print("="*80)
