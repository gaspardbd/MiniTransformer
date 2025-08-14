import argparse
import os
from typing import List, Tuple
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from decoder import LanguageModel
from utils import get_batch, estimate_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tiny decoder-only Transformer.')
    parser.add_argument('--data', type=str, default='lafontaine-baudelaire.txt', help='Path to training text file')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--max-iters', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--eval-iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--gpus', type=str, default='', help='Comma-separated GPU ids (e.g., "0,1"). Use "all" for all GPUs.')
    parser.add_argument('--n-embd', type=int, default=200)
    parser.add_argument('--n-head', type=int, default=5)
    parser.add_argument('--n-layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--generate-tokens', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=0.8)
    return parser.parse_args()


def select_device(device_arg: str, gpus_arg: str) -> Tuple[torch.device, List[int]]:
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    device_ids: List[int] = []
    if device.type == 'cuda' and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        sel = gpus_arg.strip().lower()
        if sel == 'all':
            device_ids = list(range(num_gpus))
        elif sel:
            device_ids = [int(x) for x in gpus_arg.split(',') if x.strip()]
        else:
            device_ids = [0]

        # Validate and select primary device
        for i in device_ids:
            if i < 0 or i >= num_gpus:
                raise ValueError(f"GPU id {i} is out of range [0, {num_gpus-1}]")
        device = torch.device(f'cuda:{device_ids[0]}')
        torch.cuda.set_device(device)
    return device, device_ids


def main():
    args = parse_args()

    device, device_ids = select_device(args.device, args.gpus)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training data not found at: {args.data}")

    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    if args.n_embd % args.n_head != 0:
        raise ValueError('--n-embd must be divisible by --n-head')

    model = LanguageModel(
        vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        block_size=args.block_size,
    ).to(device)

    use_dataparallel = device.type == 'cuda' and len(device_ids) > 1
    if use_dataparallel:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params_m:.2f}M parameters | vocab_size={vocab_size} | device={device}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps, train_losses, val_losses = [], [], []
    pbar = tqdm(range(1, args.max_iters + 1), total=args.max_iters, dynamic_ncols=True)
    for iter in pbar:
        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter % args.eval_interval == 0 or iter == args.max_iters:
            train_loss = estimate_loss(model, train_data, args.eval_iters, args.batch_size, args.block_size, device)
            val_loss = estimate_loss(model, val_data, args.eval_iters, args.batch_size, args.block_size, device)
            steps.append(iter)
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            pbar.set_postfix({
                'loss': f"{float(loss):.4f}",
                'train': f"{float(train_loss):.4f}",
                'val': f"{float(val_loss):.4f}",
            })

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    base_model = model.module if hasattr(model, 'module') else model
    generated = base_model.generate(context, max_new_tokens=args.generate_tokens, temperature=args.temperature)
    gen_text = decode(generated[0].tolist())
    print(gen_text)

    with open('generated_text.txt', 'w', encoding='utf-8') as f:
        f.write(gen_text)

    if steps:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, train_losses, label='Training loss')
        plt.plot(steps, val_losses, label='Validation loss')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training/Validation Loss')
        plt.tight_layout()
        plt.savefig('loss_curve.png')


if __name__ == '__main__':
    main()
