"""
  python generate_data.py --out data_mix.txt --samples 5000 --batch 8
  python generate_data.py --out data_mix.txt --samples 20000 --batch 6 --max-new-tokens 220 --quantize 4bit
"""

import os
import sys
import math
import time
import argparse
import random
from pathlib import Path
from tqdm.auto import tqdm

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-1M"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Chemin du .txt de sortie")
    p.add_argument("--samples", type=int, default=5000, help="Nombre total d'échantillons à générer")
    p.add_argument("--batch", type=int, default=6, help="Taille de batch pour la génération")
    p.add_argument("--poetry-ratio", type=float, default=0.5, help="Proportion cible de poésie [0-1]")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    p.add_argument("--temperature", type=float, default=0.9, help="Température de sampling")
    p.add_argument("--top-p", type=float, default=0.92, help="Top-p nucleus sampling")
    p.add_argument("--repetition-penalty", type=float, default=1.05, help="Pénalité de répétition")
    p.add_argument("--max-new-tokens", type=int, default=180, help="Longueur max générée par échantillon")
    p.add_argument("--quantize", type=str, default="", choices=["", "4bit"], help="Quantization 4bit via bitsandbytes")
    p.add_argument("--resume", action="store_true", help="Reprendre si le fichier existe déjà")
    p.add_argument("--flush-every", type=int, default=50, help="Flush disque tous les N échantillons")
    p.add_argument("--gpu", type=str, default="7", help="Forcer l'utilisation d'un seul GPU (ex: '7'). Laisser vide pour auto.")
    return p.parse_args()

def set_seed(seed: int):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(quantize: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Chargement du modèle: {MODEL_ID}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if quantize == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            print("bitsandbytes non installé. Installez-le ou lancez sans --quantize 4bit.", file=sys.stderr)
            raise
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        # Map entier sur le GPU 0 logique (après CUDA_VISIBLE_DEVICES)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"},
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    return tokenizer, model

# Prompts minimalistes, sans thèmes, juste le genre
POETRY_INSTRUCTIONS = [
    "Écris un poème en français. Donne un titre au poème.",
    "Compose un long poème libre en français, sur un sujet précis. Pas d'explications. Donne un titre au poème.",
    "Génère un long poème détaillé en français, concis et évocateur. Sans commentaire. Donne un titre au poème.",
]
PROSE_INSTRUCTIONS = [
    "Écris un long paragraphe de prose en français, sur un sujet précis. et ne mets pas de titre.",
    "Rédige un long paragraphe de prose en français, sur un sujet détaillé.",
    "Génère un long paragraphe complet de prose en français.",
]

def make_batch_prompts(batch_size: int, poetry_ratio: float):
    prompts = []
    genres = []
    for _ in range(batch_size):
        if random.random() < poetry_ratio:
            genres.append("POEME")
            user_content = random.choice(POETRY_INSTRUCTIONS)
        else:
            genres.append("PROSE")
            user_content = random.choice(PROSE_INSTRUCTIONS)

        # Chat format pour Qwen Instruct
        messages = [
            {"role": "system", "content": "Tu écris en français. Réponds uniquement par le texte demandé, sans balises, sans explication."},
            {"role": "user", "content": user_content},
        ]
        prompts.append(messages)
    return prompts, genres

def apply_chat_template_batch(tokenizer, batch_messages):
    # Convertit une liste de conversations en entrées texte prêtes pour generate
    inputs = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    return inputs

def decode_only_new_text(tokenizer, input_ids, output_ids):
    # Récupère uniquement la partie générée après l'invite
    gen_ids = output_ids[:, input_ids.shape[-1]:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    # Nettoyage simple: normaliser les sauts de ligne multiples
    cleaned = []
    for t in texts:
        t = t.strip()
        # Limiter un peu les répétitions d'espaces et de lignes
        t = "\n".join([line.rstrip() for line in t.splitlines()])
        cleaned.append(t)
    return cleaned

def main():
    args = parse_args()
    # Verrouiller sur un seul GPU, ex: "7" => ce GPU devient cuda:0 logique
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch  # import après éventuel CUDA_VISIBLE_DEVICES
    set_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Gestion reprise
    already = 0
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Compter les blocs séparés par des lignes vides doubles
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            already = len(blocks)
        print(f"Reprise: {already} échantillons déjà trouvés dans {out_path}")
    remaining = max(0, args.samples - already)
    if remaining == 0:
        print("Rien à faire, le fichier contient déjà le nombre ciblé d'échantillons.")
        return

    tokenizer, model = load_model(args.quantize)
    eos = tokenizer.eos_token_id

    # Ouverture en append
    f = out_path.open("a", encoding="utf-8")

    total = args.samples
    generated = already
    pbar = tqdm(total=total, initial=generated, dynamic_ncols=True)
    t0 = time.time()
    try:
        while generated < total:
            todo = min(args.batch, total - generated)
            batch_msgs, genres = make_batch_prompts(todo, args.poetry_ratio)
            inputs_text = apply_chat_template_batch(tokenizer, batch_msgs)

            enc = tokenizer(
                inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=eos,
                    eos_token_id=eos,
                )

            texts = decode_only_new_text(tokenizer, enc["input_ids"], out)

            # Écriture incrémentale: chaque bloc séparé par une ligne vide
            for txt in texts:
                # Encore un petit nettoyage pour éviter des restes de balises éventuelles
                snippet = txt.strip()
                # Eviter les blocs vides
                if not snippet:
                    continue
                f.write(snippet)
                f.write("\n\n")
                generated += 1
                pbar.update(1)

            # Progress
            if generated % args.flush_every == 0 or generated >= total:
                f.flush()
                os.fsync(f.fileno())
                elapsed = time.time() - t0
                rate = generated / max(elapsed, 1e-6)
                pbar.set_postfix({"rate": f"{rate:.2f}/s"})

    finally:
        f.close()
        pbar.close()

    print(f"Terminé. Fichier: {out_path}")
    print(f"Total écrit: {generated}")
    print("Exemples séparés par une ligne vide.")

if __name__ == "__main__":
    main()
