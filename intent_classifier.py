import argparse
import csv
import random
import re
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def clean_text(text: str) -> str:
    """Perform basic text cleaning."""
    text = re.sub(r"([,.-])", " ", text)
    text = re.sub(r"([!?()@])", r" \1 ", text)
    return text.lower()


class IntentDataset(Dataset):
    """Dataset that builds a vocabulary and encodes text and labels."""

    def __init__(self, path: str) -> None:
        self.texts: List[List[str]] = []
        self.labels: List[str] = []
        with open(path, encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                text, label = row
                self.texts.append(clean_text(text).split())
                self.labels.append(label)

        self.vocab = {"<pad>": 0, "<unk>": 1}
        for tokens in self.texts:
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

        self.label2idx = {}
        for label in self.labels:
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.encoded_texts = [
            torch.tensor([self.vocab.get(tok, 1) for tok in tokens], dtype=torch.long)
            for tokens in self.texts
        ]
        self.encoded_labels = [self.label2idx[lbl] for lbl in self.labels]

    def __len__(self) -> int:
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.encoded_texts[idx], self.encoded_labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded = [torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)]) for t in texts]
    return torch.stack(padded), torch.tensor(labels)


class SimpleClassifier(nn.Module):
    """A minimal classifier using averaged word embeddings."""

    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 50) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        mean = embeds.mean(dim=1)
        return self.fc(mean)


def train(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> None:
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, idx2label: dict) -> None:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    print(f"Accuracy: {correct / len(all_labels):.2%}")
    print("Label mapping:")
    for idx, label in idx2label.items():
        print(f"{idx}: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple deep learning intent classifier.")
    parser.add_argument("--data", default="intent_data.csv", help="Path to dataset")
    args = parser.parse_args()

    dataset = IntentDataset(args.data)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, collate_fn=collate_fn)

    model = SimpleClassifier(len(dataset.vocab), len(dataset.label2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for _ in range(5):
        train(model, train_loader, optimizer, criterion)
    evaluate(model, test_loader, dataset.idx2label)


if __name__ == "__main__":
    main()
