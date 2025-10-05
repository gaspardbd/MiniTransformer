## Mini Transformer (decoder-only)
Small decoder to generate short French fables (La Fontaine style).

### Quick start
1. Place your training text file, e.g. `lafontaine-baudelaire.txt`, in the repo root.
2. Install dependencies:
```bash
# Training + sampling
pip install torch tqdm matplotlib

# Optional: for synthetic data generation with a HF model (Qwen)
pip install transformers accelerate sentencepiece
# Optional: for 4-bit quantization in data generation
pip install bitsandbytes
```

3. (Optional) Generate synthetic data
```bash
python /shared/ssd_30T/gaspard/Mini-Transformer/generate_data.py \
  --out /shared/ssd_30T/gaspard/Mini-Transformer/data_mix.txt \
  --samples 5000 \
  --batch 8
```

4. Train on synthetic data + La Fontaine fables (two-phase)
```bash
python /shared/ssd_30T/gaspard/Mini-Transformer/train.py \
  --pretrain-data /shared/ssd_30T/gaspard/Mini-Transformer/data_mix.txt \
  --finetune-data /shared/ssd_30T/gaspard/Mini-Transformer/lafontaine-baudelaire.txt \
  --max-iters-pretrain 10000 \
  --max-iters-finetune 5000 \
  --lr-pretrain 3e-4 \
  --lr-finetune 5e-4 \
  --batch-size 16 \
  --block-size 512 \
  --eval-interval 100 \
  --eval-iters 200 \
  --device auto \
  --gpus 3 \
  --out-dir /shared/ssd_30T/gaspard/Mini-Transformer/out
```

Single-phase training example:
```bash
python /shared/ssd_30T/gaspard/Mini-Transformer/train.py \
  --data /shared/ssd_30T/gaspard/Mini-Transformer/lafontaine-baudelaire.txt \
  --max-iters 10000 \
  --device auto \
  --out-dir /shared/ssd_30T/gaspard/Mini-Transformer/out
```

5. Generate text from a saved model
```bash
python /shared/ssd_30T/gaspard/Mini-Transformer/generate.py \
  --model-dir /shared/ssd_30T/gaspard/Mini-Transformer/out \
  --prompt "Un renard et un corbeau" \
  --max-new-tokens 400 \
  --temperature 0.8 \
  --device auto
```

### Common flags (training)
- `--batch-size` (default 64)
- `--block-size` (default 256)
- `--max-iters` (default 10000)
- `--n-embd` (default 512), `--n-head` (default 8), `--n-layer` (default 24)
- `--dropout` (default 0.2)
- `--gpus` can be `"0,1"` or `all` for simple DataParallel
- `--device` can be `auto`, `cpu`, or `cuda`

Artifacts are saved in `out/` (`model.pt`, `config.json`, `generated_text.txt`, `loss_curve.png`).

### Notes
- Character-level model: the vocabulary is built from the provided training texts.
- Multi-GPU: when `--gpus` lists multiple ids (or `all`), simple `DataParallel` is used; the primary device is the first id.
- Synthetic data generator uses `Qwen/Qwen2.5-7B-Instruct-1M`. Use `--gpu <id>` to pick one GPU; `--quantize 4bit` requires `bitsandbytes`.

### DONE
- Generated synthetic poem/prose data using Qwen
- Implemented a Transformer architecture at character level
- Working training and generation code
- Added SwiGLU (as in GPT-OSS), RoPE (to get not an absolute encoding, but a relative encoding: ⟨qi′​,kj′​⟩ would depend on (i-j)) and MoE

### TO DO
- Go from character-level to token level: implement GPT tokenizer
- Larger training on a bigger pre-training dataset
- Use DDP for distributed training