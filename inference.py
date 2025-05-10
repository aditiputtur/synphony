import torch
import torch.nn as nn
import torch.nn.functional as F
from miditok import REMI, TokenizerConfig, TokSequence
from pathlib import Path
import pandas as pd
import math
from tqdm import tqdm
import argparse

# --- EXAMPLE USAGE ---
# python inference.py --genre POP --artist RICK_ASTLEY --year 1987 --output rick_roll.mid

# ─── 0. Model definition ─────────────────────────────────────────────
class RelativePositionalEncoding(nn.Module):
    """
    Sinusoidal *relative‑style* positional encoding.
    The tensor it returns has the same shape as `x`
    so you can just add it:  x + pos(x)

    Args
    ----
    d_model : int            # embedding size
    max_len : int, optional  # maximum sequence length
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create the (max_len, d_model) sinusoid table once
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)          # (L, D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it moves with .to(device)
        self.register_buffer("pe", pe)              # (L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)

        Returns
        -------
        pos : Tensor, same shape as `x`
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        # (1, L, D) – broadcast over batch dimension
        return self.pe[:seq_len].unsqueeze(0)

class TransformerDecoderBlock(nn.Module):
    """
    Decoder block that merges causal + pad masking into a (B×H, L, L) float mask,
    so no hidden bool→float blow-ups occur.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.ln1      = nn.LayerNorm(d_model)
        self.ln2      = nn.LayerNorm(d_model)
        self.ff       = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout  = nn.Dropout(dropout)

        # Precompute float causal mask: 0 on/under diag, -inf above
        causal = torch.triu(
            torch.full((max_len, max_len), float("-inf")),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self,
        x: torch.Tensor,            # (B, L, D)
        pad_mask: torch.Tensor=None  # (B, L), True=keep token, False=pad
    ) -> torch.Tensor:
        B, L, _ = x.shape
        H       = self.self_attn.num_heads
        device  = x.device
        dtype   = x.dtype

        # 1) slice the (L×L) causal mask
        causal = self.causal_mask[:L, :L]              # float32, (L, L)

        # 2) build a (B, L) float pad mask: 0 on tokens, -inf on pads
        if pad_mask is not None:
            pad_float = torch.zeros((B, L), device=device, dtype=dtype)
            pad_float = pad_float.masked_fill(~pad_mask, float("-inf"))
            # 3) expand pad_float to (B, L, L) and add causal
            #    pad_float.unsqueeze(1): (B, 1, L) → broadcast over src_len
            attn_batch = causal.unsqueeze(0) + pad_float.unsqueeze(1)  # (B, L, L)
        else:
            attn_batch = causal                               # (L, L)

        # 4) if we have a batch, repeat per-head to (B×H, L, L)
        if pad_mask is not None:
            # attn_batch: (B, L, L) → repeat each batch H times
            attn_mask = attn_batch.repeat_interleave(H, dim=0)  # (B*H, L, L)
        else:
            attn_mask = attn_batch   # 2D mask

        # 5) self-attention with ONLY attn_mask
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask
        )

        # 6) residual + norm + feed-forward + norm
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x

class Synphony(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = RelativePositionalEncoding(d_model, max_len=2048)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask=None):
        x = self.embed(x) + self.pos(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.ln(x)
        return self.out(x)

# ─── 0. Hyperparams & device ─────────────────────────────────────────────
MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_K = 8

BATCH_SIZE = 1

D_MODEL    = 128
N_LAYERS   = 1
N_HEADS    = 1

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ─── 1. Re-load tokenizer & model ────────────────────────────────────────
# (Use the same config & special tokens you trained with)
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)
# re-add the same special tokens:
genres = pd.read_csv("data/genres.csv")
titles = pd.read_csv("data/titles.csv")
artists = pd.read_csv("data/artists.csv")
years = pd.read_csv("data/years.csv")

genres_slugged = genres["slugged_genre"].values
artists_slugged = artists["slugged_artist"].values

years_vals = years["year"].values
special_toks = \
    [f"<GENRE_{g}>"  for g in genres_slugged] + \
        [f"<ARTIST_{a}>" for a in artists_slugged] + \
            [f"<YEAR_{y}>"   for y in years_vals]  + \
                ["<EOS>", "<PAD>"]

for tok in special_toks:
    tokenizer.add_to_vocab(tok)

# load the trained weights
model = Synphony(
    vocab_size=len(tokenizer),
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS
).to(device)
model.load_state_dict(torch.load("synphony_best.pt", map_location=device))
model.eval()

# helper for top-k filtering
def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    threshold = v[-1]
    return torch.where(logits < threshold, torch.full_like(logits, -float("Inf")), logits)

# ─── 2. Build conditioning prefix ────────────────────────────────────────
def build_prefix(genre, artist, year, tokenizer):
    """Convert metadata row → list[int] conditioning tokens."""
    genre_tok  = f"<GENRE_{genre}>"
    artist_tok = f"<ARTIST_{artist}>"
    year_tok   = f"<YEAR_{year}>"

    # NOTE: use tokenizer.vocab[...]  (or .token_to_id(...))
    return [
        tokenizer.vocab[genre_tok],
        tokenizer.vocab[artist_tok],
        tokenizer.vocab[year_tok],
    ]

# ─── 3. Autoregressive generation ────────────────────────────────────────
@torch.no_grad()
def generate(
    genre:str,
    artist:str,
    year:int,
    max_length:int = MAX_TOKENS
) -> list[int]:
    prefix = build_prefix(genre, artist, year, tokenizer)
    input_ids = torch.tensor([prefix], device=device)  # (1, P)
    pad_mask  = torch.ones_like(input_ids, dtype=torch.bool, device=device)

    for _ in tqdm(range(max_length - len(prefix))):
        logits = model(input_ids, pad_mask=pad_mask)
        next_logits = logits[0, -1, :]                  # (V,)
        next_logits = next_logits / TEMPERATURE
        next_logits = top_k_logits(next_logits, TOP_K)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,)
        if next_id.item() == tokenizer.vocab["<EOS>"]:
            break

        # append and extend pad_mask
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)   # (1, L+1)
        pad_mask  = torch.ones_like(input_ids, dtype=torch.bool, device=device)

    return input_ids[0].tolist()

# ─── 4. Decode to MIDI & save ────────────────────────────────────────────
def tokens_to_midi(token_ids: list[int], out_path: str):
    """
    Drop the 3 metadata tokens + optional EOS, then decode the rest.
    """
    # 1) drop the first 3 prefix IDs (genre, artist, year)
    musical_ids = token_ids[3:]
    # 2) drop trailing <EOS> if present
    eos_id = tokenizer.vocab["<EOS>"]
    if len(musical_ids) > 0 and musical_ids[-1] == eos_id:
        musical_ids = musical_ids[:-1]

    # 3) decode only the musical tokens back to a PrettyMIDI
    pm = tokenizer(musical_ids)
    # 4) write out the .mid file
    pm.dump_midi(out_path)

# ─── 5. Run it! ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a MIDI file conditioned on genre, artist, and year."
    )
    parser.add_argument(
        "--genre",
        type=str,
        required=True,
        help="Genre to condition on (e.g. POP)"
    )
    parser.add_argument(
        "--artist",
        type=str,
        required=True,
        help="Artist name to condition on (e.g. RICK_ASTLEY)"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to condition on (e.g. 1987)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="generated.mid",
        help="Path to write the generated MIDI file"
    )

    args = parser.parse_args()

    # call your existing functions
    gen_ids = generate(args.genre, args.artist, args.year)
    tokens_to_midi(gen_ids, args.output)
    print(f"🎹 Wrote MIDI to {args.output}")

if __name__ == "__main__":
    main()
