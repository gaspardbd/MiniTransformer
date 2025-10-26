import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = 2 * n_embd
        self.w = nn.Linear(n_embd, hidden)
        self.v = nn.Linear(n_embd, hidden)
        self.proj = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = swish(self.w(x))
        b = self.v(x)
        out = self.proj(a * b)
        return self.dropout(out)


class MoE(nn.Module):
    def __init__(self, n_embd, num_experts, dropout, use_swiglu=False):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(n_embd, num_experts)
        if use_swiglu:
            self.experts = nn.ModuleList([SwiGLUMLP(n_embd, dropout) for _ in range(num_experts)])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(dropout),
                ) for _ in range(num_experts)
            ])

    def forward(self, x):
        # x: (B,T,C)
        gate_logits = self.gate(x)                # (B,T,E)
        gate_weights = F.softmax(gate_logits, dim=-1)
        # Soft mixture over all experts for simplicity and stability
        out = 0
        for e in range(self.num_experts):
            expert_out = self.experts[e](x)       # (B,T,C)
            w = gate_weights[..., e].unsqueeze(-1)  # (B,T,1)
            out = out + w * expert_out
        return out


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout, rope_cos=None, rope_sin=None):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        # Register RoPE buffers so they move to the correct device automatically
        if rope_cos is not None:
            self.register_buffer('rope_cos', rope_cos)
        else:
            self.rope_cos = None
        if rope_sin is not None:
            self.register_buffer('rope_sin', rope_sin)
        else:
            self.rope_sin = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        if self.rope_cos is not None and self.rope_sin is not None:
            cos = self.rope_cos[:T, :].unsqueeze(0)
            sin = self.rope_sin[:T, :].unsqueeze(0)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        wei = torch.einsum('bth,bsh->bts', q, k) / (k.shape[-1] ** 0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))     # (B,T,T)
        wei = F.softmax(wei, dim=-1)                                     # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)                                                # (B,T,head_size)
        out = torch.bmm(wei, v)                                          # (B,T,head_size)
        return out


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout, use_rope=False):
        super().__init__()
        head_size = n_embd // num_heads
        if use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
            t = torch.arange(block_size).float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_cached = emb.cos()
            sin_cached = emb.sin()
            self.heads = nn.ModuleList([
                Head(n_embd, head_size, block_size, dropout, rope_cos=cos_cached, rope_sin=sin_cached) for _ in range(num_heads)
            ])
        else:
            self.heads = nn.ModuleList([
                Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
            ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout, use_swiglu=False):
        super().__init__()
        if use_swiglu:
            self.net = SwiGLUMLP(n_embd, dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, use_moe=False, num_experts=4, use_rope=False, use_swiglu=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, use_rope=use_rope)
        self.ffwd = MoE(n_embd, num_experts, dropout, use_swiglu=use_swiglu) if use_moe else FeedForward(n_embd, dropout, use_swiglu=use_swiglu)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))            # (B,T,C)
        x = x + self.ffwd(self.ln2(x))          # (B,T,C)
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=200,
        n_head=5,
        n_layer=4,
        dropout=0.2,
        block_size=128,
        use_moe=False,
        num_experts=4,
        use_rope=False,
        use_swiglu=False,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, use_moe=use_moe, num_experts=num_experts, use_rope=use_rope, use_swiglu=use_swiglu) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                                 # (B,T,C)
        if self.use_rope:
            x = tok_emb
        else:
            pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T,C)
            x = tok_emb + pos_emb                                               # (B,T,C)
        x = self.blocks(x)                                                  # (B,T,C)
        x = self.ln_f(x)                                                    # (B,T,C)
        logits = self.head(x)                                               # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, self.vocab_size)  # (B*T, vocab_size)
            targets = targets.view(-1)                  # (B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
