import preprocess
import NewDataPreprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import tensorflow as tf
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#filepath = "data.csv"
trainpath = "train.csv"
testpath = "test.csv"

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/home/public/mdeweese/SocialProj/c/hf_cache")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, cache_dir="/home/public/mdeweese/SocialProj/c1/hf_cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the datasets
train = NewDataPreprocess.get_dataset(trainpath, tokenizer)
test = NewDataPreprocess.get_dataset(testpath, tokenizer)
#train, test = preprocess.get_datasets_split(filepath, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results/split", 
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_metrics
)


trainer.train()

model_save_path = "./saved_model"  # Path to save the model
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")



    
results = trainer.evaluate()
print(results)


