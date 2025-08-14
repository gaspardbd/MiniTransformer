## Mini Transformer (decoder-only)
Small decoder to generate short French fables (La Fontaine style).

### Quick start
1. Place your training text file, e.g. `lafontaine-baudelaire.txt`, in the repo root.
2. Install dependencies:
```bash
pip install torch matplotlib
```
3. Train:
```bash
python train.py --data lafontaine-baudelaire.txt --device auto --gpus all
```

Common flags:
- `--batch-size` (default 64)
- `--block-size` (default 256)
- `--max-iters` (default 200)
- `--n-embd` (default 200), `--n-head` (default 5), `--n-layer` (default 4)
- `--gpus` can be `"0,1"` or `all` for simple DataParallel

Artifacts will be saved to `generated_text.txt` and `loss_curve.png`.
