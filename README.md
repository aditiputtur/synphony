# SynPhony

> A transformer-based symbolic music generation system that composes 
> original pieces conditioned on genre, artist, and era.

[Read the full writeup on Medium](https://medium.com/@aditi.puttur/synphony-transcending-musical-eras-and-genres-with-neural-networks-e18db3741cbf)

---

## What It Does

SynPhony generates symbolic music (MIDI) that reflects a specific 
musical style. You choose:

- **Genre**: Classical, Jazz, Pop, Rock, Electronic, and 10 more
- **Era**: Any decade from 1945–2010
- **Artist**: 2,956 artists including Frank Sinatra, Lady Gaga, 
  Coldplay, and more

The model composes a piece token by token, guided by conditioning 
tokens embedded at the start of each sequence, then converts the 
output back to a playable MIDI file.

---

## Dataset

Three sources aligned into 31,034 fully matched records:

| Source | Contents |
|---|---|
| Lakh MIDI Dataset (LMD) | 31,034 unique MIDI tracks (from 116,189 total) |
| Million Song Dataset (MSD) | Matched metadata (.h5): tempo, key, artist |
| Tagtraum Genre Tags | 191,401 genre labels matched to LMD tracks |

After tokenization and filtering: **6,150 training sequences (~3 GB)**

---

## Model Architecture

Decoder-only transformer trained with next-token prediction 
(teacher forcing):

| Component | Configuration |
|---|---|
| Vocabulary | 3,534 tokens (notes, chords, timing, 125 conditioning tokens) |
| Embedding | 768-dimensional vector space |
| Positional encoding | Relative sinusoidal, up to 1,024 tokens |
| Transformer blocks | 8 decoder blocks |
| Attention | 12-head self-attention, 64 dim/head |
| Regularization | Dropout, label smoothing (0.1), gradient clipping (‖g‖₂ ≤ 1) |
| Optimizer | AdamW (lr=3e-4, weight decay=1e-2) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2, min lr=1e-6) |
| Hardware | NVIDIA L4 GPU (Google Cloud g2-standard-8) |
| Training time | ~7 hours / 50 epochs |

---

## Experiments & Results

Four progressive experiments scaling architecture, data, and 
optimization:

| Experiment | Key Changes | Val Perplexity |
|---|---|---|
| 1 | Baseline, 10 epochs | High |
| 2 | More layers/heads, 50 epochs, larger batch | Significant drop |
| 3 | ReduceLROnPlateau added | ~3.0 (plateau broken) |
| 4 | D_MODEL=768, deeper/wider, context=1024, batch=8 | **2.43** ✅ |

Best result: **2.43 validation perplexity** — the model reliably 
predicts the next musical token in a stylistically conditioned sequence.

---

## Tech Stack
```
PyTorch 2.3 · Python 3.12 · MIDI Processing (pretty_midi, librosa)  
Hugging Face (tokenizer) · Streamlit (UI) · Google Cloud (training)
```

---

## Repo Structure
```
synphony/
├── synphony.ipynb          # Full training pipeline
├── inference.py            # Standalone generation (no UI required)
├── streamlit_app2.py       # Web UI (requires local model checkpoint)
├── hdf5_getters.py         # MSD metadata extraction utilities
├── synphony_best.pt        # Best model checkpoint (Experiment 4)
├── requirements.txt
└── packages.txt
```

---

## Running Locally
```bash
git clone https://github.com/aditiputtur/synphony
cd synphony
pip install -r requirements.txt

# Generate music via command line (no UI needed)
python inference.py --genre jazz --artist "frank sinatra" --year 1955

# Run the Streamlit UI (requires local environment setup)
streamlit run streamlit_app2.py
```

> **Note:** The Streamlit UI requires a local Python environment with 
> system audio dependencies. See `packages.txt` for system-level 
> requirements. The `inference.py` script works without the UI.

---

## Reproducibility

- Hardware: NVIDIA L4 GPU
- Framework: PyTorch 2.3, Python 3.12
- Runtime: 6h 54m for 50 epochs
- Random seed: 42
- Environment lock file included in `requirements.txt`
