import os
import json
import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

# ---------------------------------------------------------
# 1) Simple tokenizer
# ---------------------------------------------------------
class SimpleTokenizer:
    """
    Very small tokenizer for text training:
    - character-level vocabulary
    - good for learning and testing
    - easy to replace with BPE later
    """
    def __init__(self, text=None):
        self.vocab = ["<PAD>", "<UNK>"]
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text):
        chars = sorted(set(text))
        for ch in chars:
            if ch not in self.stoi:
                idx = len(self.vocab)
                self.vocab.append(ch)
                self.stoi[ch] = idx
                self.itos[idx] = ch

    def encode(self, text):
        return [self.stoi.get(ch, self.stoi["<UNK>"]) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "<UNK>") for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.stoi = {ch: i for i, ch in enumerate(tok.vocab)}
        tok.itos = {i: ch for i, ch in enumerate(tok.vocab)}
        return tok


# ---------------------------------------------------------
# 2) GPT-like model
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == embed_dim, "embed_dim must be divisible by n_head"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_head, T, T]

        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_head=8, n_layer=6, seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, embed_dim * 4, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_embedding(x)
        pos = self.position_embedding(torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, input_ids, max_new_tokens=100, temperature=0.9):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.shape[1] > self.seq_len:
                    input_ids = input_ids[:, -self.seq_len:]
                logits = self(input_ids)
                logits = logits[:, -1, :] / max(temperature, 1e-8)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ---------------------------------------------------------
# 3) Training functions
# ---------------------------------------------------------
def load_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_train_chunks(tokenizer, text, seq_len):
    ids = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len):
        x = ids[i:i + seq_len]
        y = ids[i + 1:i + seq_len + 1]
        if len(x) == seq_len and len(y) == seq_len:
            chunks.append((x, y))
    return chunks


def train_model(model, tokenizer, text, epochs=5, batch_size=8, seq_len=128, lr=3e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.stoi["<PAD>"]

    chunks = make_train_chunks(tokenizer, text, seq_len)
    if len(chunks) == 0:
        raise ValueError("Text is too short for training. Use longer text or smaller seq_len.")

    for epoch in range(epochs):
        model.train()
        random.shuffle(chunks)
        total_loss = 0.0
        steps = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            max_len = max(len(x) for x, _ in batch)

            xb = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            yb = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)

            for j, (x, y) in enumerate(batch):
                xb[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=device)
                yb[j, :len(y)] = torch.tensor(y, dtype=torch.long, device=device)

            logits = model(xb)
            logits = logits.view(-1, model.vocab_size)
            targets = yb.view(-1)

            loss = F.cross_entropy(logits, targets, ignore_index=pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch + 1}/{epochs} | loss: {avg_loss:.4f}")

    return model


def save_model(model, tokenizer, model_path="model.pt", tokenizer_path="tokenizer.json"):
    torch.save(model.state_dict(), model_path)
    tokenizer.save(tokenizer_path)
    print(f"Saved model to {model_path}")
    print(f"Saved tokenizer to {tokenizer_path}")


def load_model(model_class, model_path, tokenizer_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    tok = SimpleTokenizer.load(tokenizer_path)
    model = model_class(vocab_size=len(tok.vocab), embed_dim=256, n_head=8, n_layer=6, seq_len=128)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tok


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.9, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    encoded = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(encoded, max_new_tokens=max_new_tokens, temperature=temperature)
    text = tokenizer.decode(generated[0].cpu().tolist())
    return text


# ---------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small GPT-style AI training script")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--data", type=str, default="data.txt", help="Text file for training")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Training data not found: {args.data}")
        text = load_text(args.data)
        tokenizer = SimpleTokenizer(text)
        model = TinyGPT(vocab_size=len(tokenizer.vocab), embed_dim=256, n_head=8, n_layer=6, seq_len=args.seq_len)
        model = train_model(model, tokenizer, text, epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr, device=device)
        save_model(model, tokenizer, "model.pt", "tokenizer.json")
        print("Training complete.")
    
    if args.generate:
        if not os.path.exists("model.pt") or not os.path.exists("tokenizer.json"):
            print("Model not found. Train first with --train")
        else:
            model, tokenizer = load_model(TinyGPT, "model.pt", "tokenizer.json", device=device)
            out = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.9, device=device)
            print("\n=== GENERATED TEXT ===")
            print(out)
            print("=====================")
