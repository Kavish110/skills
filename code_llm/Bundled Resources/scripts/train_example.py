"""
Minimal PyTorch training example for a tiny transformer-style decoder model.
This script is config-driven to keep parameter changes in YAML instead of code.

Run: python train_example.py --config "../assets/example_config.yaml"
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml


class DummyDataset(Dataset):
    def __init__(self, vocab_size=1000, seq_len=32, size=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x[:-1], x[1:]


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.tok_emb(x).mean(dim=1)
        return self.lin(h)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optim, loss_fn, device):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb[:, 0])
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def save_checkpoint(model, output_dir, checkpoint_name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, checkpoint_name)
    torch.save({'model_state': model.state_dict()}, path)
    return path


def main():
    parser = argparse.ArgumentParser(description='Train a tiny decoder model from config.')
    parser.add_argument('--config', type=str, default='../assets/example_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    seq_len = config['data']['seq_len']
    dataset_size = config['training'].get('dataset_size', 1000)
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']

    ds = DummyDataset(vocab_size=vocab_size, seq_len=seq_len, size=dataset_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = TinyDecoder(vocab_size=vocab_size, d_model=d_model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train_one_epoch(model, dl, optim, loss_fn, device)
        print(f"Epoch {epoch} loss: {loss:.4f}")

    checkpoint_path = save_checkpoint(model, config['checkpoint']['output_dir'], config['checkpoint']['checkpoint_name'])
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()
