# If venv not initialised : python -m venv .venv
# .venv/bin/pip install matplotlib torch transformers datasets scikit-learn pandas accelerate -U
#source .venv/bin/activate
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_recall_curve
import shap
import logging, sys, tempfile, shutil
import matplotlib.pyplot as plt

FILE_NAME = 'corpus_ANNOTATED_TRANSLATED.csv'
TEXT_COLUMN = 'text'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'

# parameters
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
N_SPLITS = 5
RANDOM_STATE = 42

resfile = ('results/scibert_epochs'+str(NUM_TRAIN_EPOCHS)+'_lrate'+
           str(LEARNING_RATE)+'_batch'+str(BATCH_SIZE)+'_seqlength'+str(MAX_SEQ_LENGTH)+
           '_folds'+str(N_SPLITS)+'_seed'+str(RANDOM_STATE))
logfile = resfile+'.txt'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(logfile, mode='w') # 'w' overwrites, 'a' appends
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

temp_run_dir = tempfile.mkdtemp()

# Load data
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




kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

acc_scores = []
micro_f1_scores = []
macro_f1_scores = []
hamming_losses = []
best_thresholds = []

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    logger.info(f"\n\n\n" + "="*60)
    logger.info(f"--- Fine-Tuning SciBERT: Fold {fold + 1}/{N_SPLITS} ---")

    train_dataset = tokenized_hg_dataset.select(train_index)
    test_dataset = tokenized_hg_dataset.select(test_index)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=temp_run_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        logging_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=RANDOM_STATE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer#,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # find optimal threshold
    val_predictions = trainer.predict(test_dataset)
    logits = val_predictions.predictions
    probs = nn.functional.sigmoid(torch.tensor(logits)).numpy()
    y_true = np.array(test_dataset["labels"])
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), probs.ravel())
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (probs >= best_threshold).astype(int)

    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1_micro = f1_score(y_true, y_pred, average='micro')
    fold_f1_macro = f1_score(y_true, y_pred, average='macro')
    fold_hamming = hamming_loss(y_true, y_pred)
    logger.info( f"Fold {fold + 1} | Acc: {fold_acc:.3f} | Micro-F1: {fold_f1_micro:.3f} | Macro-F1: {fold_f1_macro:.3f} | Hamming: {fold_hamming:.3f}")
    acc_scores.append(fold_acc)
    micro_f1_scores.append(fold_f1_micro)
    macro_f1_scores.append(fold_f1_macro)
    hamming_losses.append(fold_hamming)
    best_thresholds.append(best_threshold)


# SHAP interpretation for a sample in the last fold
def model_predict(texts):
    encoded = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**encoded)
    return torch.sigmoid(outputs.logits).cpu().numpy()

explainer = shap.Explainer(model_predict, tokenizer)
shap_values = explainer(df.iloc[test_index]['text'].iloc[:5].values)
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[0, :, 0], show=False) # Explain the first label for the first paper in the sample
plt.title(f"SHAP Importance: {label_cols[0]}")
plt.savefig(resfile+".png", bbox_inches='tight')
plt.close()


# Output metrics

logger.info("\n\n\n" + "="*80)
logger.info("SciBERT Cross-Validation Complete. Average Scores:")

metrics = {
    "Accuracy": acc_scores,
    "Micro-F1":          micro_f1_scores,
    "Macro-F1":          macro_f1_scores,
    "Hamming Loss":      hamming_losses
}

logger.info("="*40)
logger.info(f"{'Metric':<20} | {'Mean':<8} | {'Std Dev':<8}")
logger.info("-"*40)

for name, scores in metrics.items():
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    logger.info(f"{name:<20} | {mean_val:.4f}   | {std_val:.4f}")

logger.info("="*40)

logger.info(f"Optimal Threshold used in last fold: {np.mean(best_thresholds):.4f} +-{np.std(best_thresholds):.4f} ")

shutil.rmtree(temp_run_dir)



