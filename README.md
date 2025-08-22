## Mini Transformer (decoder-only)
Small decoder to generate short French fables (La Fontaine style).

### Quick start
1. Place your training text file, e.g. `lafontaine-baudelaire.txt`, in the repo root.
2. Install dependencies:
```bash
pip install torch matplotlib
```

3. Generate some synthetic data
```bash
python generate_data.py --out data_mix.txt --samples 5000 --batch 8

```
3. Train on synthetic data + lafontaine fables
```bash
python /shared/ssd_30T/gaspard/Mini-Transformer/train.py   --pretrain-data /shared/ssd_30T/gaspard/Mini-Transformer/data_mix.txt   --finetune-data /shared/ssd_30T/gaspard/Mini-Transformer/lafontaine-baudelaire.txt   --max-iters-pretrain 10000   --max-iters-finetune 5000   --lr-pretrain 3e-4   --lr-finetune 5e-4   --batch-size 16   --block-size 512   --eval-interval 1
00   --eval-iters 200   --device auto   --gpus 3   --out-dir /shared/ssd_30T/gaspard/Mini-Transformer/out
```

Common flags:
- `--batch-size` (default 64)
- `--block-size` (default 256)
- `--max-iters` (default 200)
- `--n-embd` (default 200), `--n-head` (default 5), `--n-layer` (default 4)
- `--gpus` can be `"0,1"` or `all` for simple DataParallel

Artifacts will be saved to `generated_text.txt` and `loss_curve.png`.
