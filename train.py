import argparse
import json
import os
from typing import Dict, List, Tuple
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from decoder import LanguageModel
from utils import get_batch, estimate_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tiny decoder-only Transformer.')
    parser.add_argument('--data', type=str, default='lafontaine-baudelaire.txt', help='Path to training text file (single-phase)')
    parser.add_argument('--pretrain-data', type=str, default='', help='Path to large dataset for phase 1 (pretraining)')
    parser.add_argument('--finetune-data', type=str, default='', help='Path to smaller dataset for phase 2 (finetuning)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--max-iters', type=int, default=10000)
    parser.add_argument('--max-iters-pretrain', type=int, default=0, help='Overrides --max-iters for phase 1 if > 0')
    parser.add_argument('--max-iters-finetune', type=int, default=0, help='Overrides --max-iters for phase 2 if > 0')
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--eval-iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-pretrain', type=float, default=3e-4, help='Learning rate for phase 1')
    parser.add_argument('--lr-finetune', type=float, default=5e-4, help='Learning rate for phase 2')
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--gpus', type=str, default='', help='Comma-separated GPU ids (e.g., "0,1"). Use "all" for all GPUs.')
    parser.add_argument('--n-embd', type=int, default=512)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-layer', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use-moe', action='store_true', help='Enable Mixture-of-Experts FFN')
    parser.add_argument('--num-experts', type=int, default=4, help='Number of experts for MoE')
    parser.add_argument('--use-rope', action='store_true', help='Enable RoPE in attention')
    parser.add_argument('--use-swiglu', action='store_true', help='Use SwiGLU MLP instead of GELU MLP')
    parser.add_argument('--generate-tokens', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--out-dir', type=str, default='out', help='Directory to save model and artifacts')
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


def build_vocab_from_texts(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars_set = set()
    for t in texts:
        chars_set.update(t)
    chars = sorted(list(chars_set))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def main():
    args = parse_args()

    device, device_ids = select_device(args.device, args.gpus)

    # Determine training mode (single-phase or two-phase)
    two_phase = bool(args.pretrain_data and args.finetune_data)

    texts_for_vocab: List[str] = []
    if two_phase:
        if not os.path.exists(args.pretrain_data):
            raise FileNotFoundError(f"Pretraining data not found at: {args.pretrain_data}")
        if not os.path.exists(args.finetune_data):
            raise FileNotFoundError(f"Finetuning data not found at: {args.finetune_data}")
        with open(args.pretrain_data, 'r', encoding='utf-8') as f:
            pre_text = f.read()
        with open(args.finetune_data, 'r', encoding='utf-8') as f:
            ft_text = f.read()
        texts_for_vocab = [pre_text, ft_text]
    else:
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Training data not found at: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            single_text = f.read()
        texts_for_vocab = [single_text]

    stoi, itos = build_vocab_from_texts(texts_for_vocab)
    vocab_size = len(stoi)
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    # Build datasets
    if two_phase:
        data_pre = torch.tensor(encode(pre_text), dtype=torch.long)
        n_pre = int(0.9 * len(data_pre))
        train_pre = data_pre[:n_pre]
        val_pre = data_pre[n_pre:]

        data_ft = torch.tensor(encode(ft_text), dtype=torch.long)
        n_ft = int(0.9 * len(data_ft))
        train_ft = data_ft[:n_ft]
        val_ft = data_ft[n_ft:]
    else:
        data = torch.tensor(encode(single_text), dtype=torch.long)
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
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        use_rope=args.use_rope,
        use_swiglu=args.use_swiglu,
    ).to(device)

    use_dataparallel = device.type == 'cuda' and len(device_ids) > 1
    if use_dataparallel:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params_m:.2f}M parameters | vocab_size={vocab_size} | device={device}")

    os.makedirs(args.out_dir, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=(args.lr_pretrain if two_phase else args.lr), weight_decay=args.weight_decay)

    steps, train_losses, val_losses = [], [], []

    def run_phase(phase_name: str, max_iters: int, lr: float, train_tensor: torch.Tensor, val_tensor: torch.Tensor, start_step: int) -> int:
        for g in optimizer.param_groups:
            g['lr'] = lr
        pbar = tqdm(range(1, max_iters + 1), total=max_iters, dynamic_ncols=True)
        for iter in pbar:
            xb, yb = get_batch(train_tensor, args.batch_size, args.block_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step = start_step + iter
            if iter % args.eval_interval == 0 or iter == max_iters:
                train_loss = estimate_loss(model, train_tensor, args.eval_iters, args.batch_size, args.block_size, device)
                val_loss = estimate_loss(model, val_tensor, args.eval_iters, args.batch_size, args.block_size, device)
                steps.append(global_step)
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))
                pbar.set_postfix({
                    'phase': phase_name,
                    'loss': f"{float(loss):.4f}",
                    'train': f"{float(train_loss):.4f}",
                    'val': f"{float(val_loss):.4f}",
                    'lr': lr,
                })
        return start_step + max_iters

    # Training
    total_steps = 0
    if two_phase:
        iters_pre = args.max_iters_pretrain if args.max_iters_pretrain > 0 else args.max_iters
        total_steps = run_phase('pretrain', iters_pre, args.lr_pretrain, train_pre, val_pre, total_steps)
        iters_ft = args.max_iters_finetune if args.max_iters_finetune > 0 else args.max_iters
        total_steps = run_phase('finetune', iters_ft, args.lr_finetune, train_ft, val_ft, total_steps)
    else:
        iters_single = args.max_iters
        total_steps = run_phase('train', iters_single, args.lr, train_data, val_data, total_steps)

    # Save model and artifacts
    base_model = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(args.out_dir, 'model.pt')
    torch.save(base_model.state_dict(), save_path)

    # Save config including vocabulary mapping
    config = {
        'vocab_size': vocab_size,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'n_layer': args.n_layer,
        'dropout': args.dropout,
        'block_size': args.block_size,
        'use_moe': args.use_moe,
        'num_experts': args.num_experts,
        'use_rope': args.use_rope,
        'use_swiglu': args.use_swiglu,
        'itos': [itos[i] for i in range(vocab_size)],
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w', encoding='utf-8') as cf:
        json.dump(config, cf, ensure_ascii=False, indent=2)

    # Quick sample generation to file
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = base_model.generate(context, max_new_tokens=args.generate_tokens, temperature=args.temperature)
    gen_text = decode(generated[0].tolist())
    print(gen_text)

    with open(os.path.join(args.out_dir, 'generated_text.txt'), 'w', encoding='utf-8') as f:
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
        plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))


if __name__ == '__main__':
    main()
