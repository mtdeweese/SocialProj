import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

def get_lists(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["content", "label"])
    texts, _, labels, _ = train_test_split(
        df["content"].tolist(),
        df["label"].tolist(),
        test_size=0.01,
        random_state=42
    )
    return texts, labels



def tokenize(content, tokenizer):
    encodings = tokenizer(content, truncation=True, padding=True, max_length=512)
    return encodings


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, original_texts=None):
        self.encodings = encodings
        self.labels = labels
        self.original_texts = original_texts
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.original_texts:
            item["content"] = self.original_texts[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

def get_dataset(filepath, tokenizer):
    texts, labels = get_lists(filepath)
    encodings = tokenize(texts, tokenizer)
    dataset = RedditDataset(encodings, labels, original_texts=texts)
    print(f"Dataset size: {len(dataset)}")
    return dataset



