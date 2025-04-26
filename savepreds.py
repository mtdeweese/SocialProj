import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import preprocess
import os
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

filepath = "data.csv"
original_df = pd.read_csv(filepath)
train_dataset, test_dataset = preprocess.get_datasets(filepath, tokenizer)


test_loader = DataLoader(test_dataset, batch_size=64)
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)


test_size = len(test_dataset)
original_test_df = original_df.tail(test_size).copy()
original_test_df["predictions"] = all_preds


original_test_df.to_csv("test_predictions.csv", index=False)
print("? Predictions saved to 'test_predictions.csv'")