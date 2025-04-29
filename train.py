import preprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import tensorflow as tf
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

filepath= "data.csv"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/home/public/mdeweese/SocialProj/c/hf_cache")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, cache_dir="/home/public/mdeweese/SocialProj/c1/hf_cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train, test = preprocess.get_datasets(filepath, tokenizer)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
)

trainer.train()