import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import preprocess
import os
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from scipy.special import softmax
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = "./saved_model"  # Replace this with the path to your saved model

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/home/public/mdeweese/SocialProj/c/hf_cache")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, cache_dir="/home/public/mdeweese/SocialProj/c1/hf_cache")

# Set device to CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

filepath = "data.csv"
_, test = preprocess.get_datasets_split(filepath, tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir="./results",  # Directory where results will be saved
    per_device_eval_batch_size=64,  # Batch size for evaluation
)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

results = trainer.evaluate()
print("Evaluation Results:", results)



preds = trainer.predict(test)
pred_labels = preds.predictions.argmax(-1) 
true_labels = preds.label_ids

print(classification_report(true_labels, pred_labels, digits=4))



test_loader = DataLoader(test, batch_size=16)


all_preds = []
all_probs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = softmax(logits.cpu().numpy(), axis=-1)
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_probs.extend(probs[:, 1])


test_df = pd.DataFrame({
    "label": true_labels,
    "prediction": all_preds,
    "prob_1": all_probs
})
test_df["prediction"] = all_preds
test_df["prob_1"] = all_probs


test_df.to_csv("predictions_without_trainer.csv", index=False)


print(classification_report(test_df["label"], test_df["prediction"]))