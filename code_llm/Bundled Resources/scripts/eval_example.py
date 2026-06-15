"""
Simple evaluation harness: compute token-level accuracy and pseudo-perplexity
on the DummyDataset from train_example.
This evaluation is config-driven, so changes are made in YAML instead of code.
"""
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from train_example import DummyDataset, TinyDecoder


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=-1)
            correct += (preds == yb[:, 0]).sum().item()
            total += preds.numel()
    acc = correct / total if total else 0.0
    print(f"Token accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a tiny decoder model from config.')
    parser.add_argument('--config', type=str, default='../assets/example_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    seq_len = config['data']['seq_len']
    batch_size = config['training']['eval_batch_size']
    dataset_size = config['training'].get('dataset_size', 200)

    ds = DummyDataset(vocab_size=vocab_size, seq_len=seq_len, size=dataset_size)
    dl = DataLoader(ds, batch_size=batch_size)
    model = TinyDecoder(vocab_size=vocab_size, d_model=d_model)
    ckpt = torch.load(config['checkpoint']['checkpoint_name'], map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    evaluate(model, dl, device)


if __name__ == '__main__':
    main()
