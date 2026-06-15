"""
Simple inference latency/throughput micro-benchmark for a loaded model.
Changes in benchmark behavior are configured in YAML rather than code.
"""
import argparse
import time
import torch
import yaml
from train_example import TinyDecoder


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def benchmark(model, device, seq_len=32, batch_size=1, iters=50, warmup_steps=5):
    model.eval()
    inp = torch.randint(0, model.tok_emb.num_embeddings, (batch_size, seq_len)).to(device)
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(inp)
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.time()
            _ = model(inp)
            t1 = time.time()
            times.append(t1 - t0)
    avg = sum(times) / len(times)
    print(f"Avg latency (s): {avg:.6f}, Throughput tok/s: {batch_size*seq_len/avg:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference speed using YAML config.')
    parser.add_argument('--config', type=str, default='../assets/example_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']

    model = TinyDecoder(vocab_size=vocab_size, d_model=d_model).to(device)
    benchmark(
        model,
        device,
        seq_len=config['benchmark']['seq_len'],
        batch_size=config['benchmark']['batch_size'],
        iters=config['benchmark']['iters'],
        warmup_steps=config['benchmark']['warmup_steps'],
    )


if __name__ == '__main__':
    main()
