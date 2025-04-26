import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

def get_split(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["text", "label"])
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )
    return train_texts, test_texts, train_labels, test_labels

def get_topic_split(filepath):
    import random
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["text", "label"])

    label_0_df = df[df["label"] == 0]
    label_1_df = df[df["label"] == 1]

    label_0_topics = label_0_df["topic"].unique().tolist()
    label_1_topics = label_1_df["topic"].unique().tolist()


    random.seed(42)

    label_0_test_topics = random.sample(label_0_topics, k=2)
    label_1_test_topics = random.sample(label_1_topics, k=int(0.2 * len(label_1_topics))) 

    test_df = df[
        ((df["label"] == 0) & (df["topic"].isin(label_0_test_topics))) |
        ((df["label"] == 1) & (df["topic"].isin(label_1_test_topics)))
    ]
    train_df = df.drop(test_df.index)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    return train_texts, test_texts, train_labels, test_labels


def tokenize(train, test, tokenizer):
    train_encodings = tokenizer(train, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test, truncation=True, padding=True, max_length=512)
    return train_encodings, test_encodings


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, original_texts=None):
        self.encodings = encodings
        self.labels = labels
        self.original_texts = original_texts
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.original_texts:
            item["text"] = self.original_texts[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

def get_datasets(filepath, tokenizer):
    train_t, test_t, train_l, test_l = get_split(filepath)
    train_encodings, test_encodings = tokenize(train_t, test_t, tokenizer)
    train_dataset = RedditDataset(train_encodings, train_l, original_texts=train_t)
    test_dataset = RedditDataset(test_encodings, test_l, original_texts=test_t)
    return train_dataset, test_dataset

def get_datasets_split(filepath, tokenizer):
    train_t, test_t, train_l, test_l = get_topic_split(filepath)
    train_encodings, test_encodings = tokenize(train_t, test_t, tokenizer)
    train_dataset = RedditDataset(train_encodings, train_l, original_texts=train_t)
    test_dataset = RedditDataset(test_encodings, test_l, original_texts=test_t)
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset


