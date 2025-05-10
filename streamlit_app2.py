import streamlit as st
import pandas as pd
import pretty_midi
import tempfile
import wave
import random
import io
import unicodedata
from pathlib import Path
from pydub import AudioSegment

from inference import generate, tokens_to_midi

# ─── NORMALIZATION HELPER ────────────────────────────────────
def normalize_name(s: str) -> str:
    s_nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s_nfkd if not unicodedata.combining(c)).lower().strip()

# ─── CONFIG & STYLING ────────────────────────────────────────
st.set_page_config(page_title="SynPhony Music Generator", layout="wide")
st.markdown("""
<style>
  .main { background-color: #6EADBC; }
  .stTitle, .stSubheader { color: #000 !important; }

  /* Buttons */
  div.stButton > button, div.stDownloadButton > button {
      background-color: #0B4B5A; color: white;
      border: 2px solid transparent; border-radius: 8px;
      padding: 0.5em 1em; font-weight: 600;
      transition: all 0.3s ease-in-out;
  }
  div.stButton > button:hover, div.stDownloadButton > button:hover {
      border: 2px solid #d9fff8; color: #d9fff8;
      background-color: #0B4B5A;
  }
  div.stButton > button:active, div.stDownloadButton > button:active {
      border: 2px solid #d9fff8 !important;
      color: #d9fff8 !important;
      background-color: #083d4a !important;
  }

  /* Selectboxes */
  div[data-baseweb="select"] > div {
      border: 2px solid #0B4B5A !important;
      border-radius: 6px !important;
      transition: all 0.3s ease-in-out;
  }
  div[data-baseweb="select"]:hover > div,
  div[data-baseweb="select"]:focus-within > div {
      border: 2px solid #d9fff8 !important;
      box-shadow: none !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────
st.image("SynPhony.png", width=300)
st.title("SynPhony: Music Your Way")
st.subheader("Build a unique musical piece by selecting an artist, an era, and a genre!")

# ─── LOAD METADATA ─────────────────────────────────────────
genre_meta  = pd.read_csv("data/genres.csv")   # columns: genre, slugged_genre
artist_meta = pd.read_csv("data/artists.csv")  # columns: artist, slugged_artist
year_meta   = pd.read_csv("data/years.csv")    # column: year

# Build lookup dicts keyed by normalized name
artist_map = {
    normalize_name(row.artist): row.slugged_artist
    for _, row in artist_meta.iterrows()
}
genre_map = {
    normalize_name(row.genre): row.slugged_genre
    for _, row in genre_meta.iterrows()
}

# ─── HARD-CODED UI LISTS ─────────────────────────────────────
artists = [
    "Britney Spears", "Elvis Presley", "Whitney Houston",
    "Coldplay", "Frank Sinatra", "Lady Gaga",
    "Beyoncé", "Brandy", "Radiohead"
]
genres = genre_meta["genre"].tolist()
years  = year_meta["year"].astype(int).tolist()
decades = sorted({(y // 10)*10 for y in years})
eras    = [f"{d}s" for d in decades if d > 0]

# ─── UI CONTROLS ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    composer = st.selectbox("Select an Artist", [""] + artists)
with c2:
    era      = st.selectbox("Select an Era", [""] + eras)
with c3:
    genre    = st.selectbox("Select a Genre", [""] + genres)

# ─── GENERATION ──────────────────────────────────────────────
if st.button("Generate Music") and composer and era and genre:
    # clear old outputs
    for k in ("midi_bytes", "wav_bytes", "mp3_bytes"):
        st.session_state.pop(k, None)

    st.success(f"Generating music in the style of **{composer}** from the **{era}** in **{genre}**…")

    # 1) Normalize & map to slugs
    n_comp = normalize_name(composer)
    n_genre= normalize_name(genre)
    slug_a = artist_map.get(n_comp)
    slug_g = genre_map.get(n_genre)
    if not slug_a or not slug_g:
        st.error("Selected artist or genre not found in metadata.")
        st.stop()

    # 2) Pick a random year in the decade
    ds  = int(era.rstrip("s"))
    yrs = [y for y in years if ds <= y < ds+10]
    if not yrs:
        st.error(f"No training data for the {era}.")
        st.stop()
    rep_y = random.choice(yrs)

    # 3) Generate tokens → MIDI
    token_ids = generate(slug_g, slug_a, rep_y)
    midi_path = f"syn_{slug_a}_{rep_y}_{slug_g}.mid"
    tokens_to_midi(token_ids, midi_path)

    # 4) Synthesize PCM via pretty_midi
    pm  = pretty_midi.PrettyMIDI(midi_path)
    pcm = pm.fluidsynth(fs=16000)  # numpy array of int16 samples
    st.session_state["pcm"] = pcm

    # 5) Wrap into a valid WAV file
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_tmp.close()
    with wave.open(wav_tmp.name, 'wb') as wf:
        nch = 1 if pcm.ndim == 1 else pcm.shape[1]
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    wav_bytes = Path(wav_tmp.name).read_bytes()

    # 6) Convert WAV → MP3 in-memory
    audio_seg = AudioSegment.from_file(wav_tmp.name, format="wav")
    mp3_io    = io.BytesIO()
    audio_seg.export(mp3_io, format="mp3")
    mp3_bytes = mp3_io.getvalue()

    # 7) Store outputs
    st.session_state["midi_bytes"] = Path(midi_path).read_bytes()
    st.session_state["wav_bytes"]  = wav_bytes
    st.session_state["mp3_bytes"]  = mp3_bytes

# ─── DOWNLOAD & PLAYBACK ─────────────────────────────────────
left, right = st.columns(2)
with left:
    if "midi_bytes" in st.session_state:
        st.download_button(
            "Download MIDI File",
            data=st.session_state["midi_bytes"],
            file_name="generated_music.mid",
            mime="audio/midi",
            key="dl_midi",
        )
with right:
    if "mp3_bytes" in st.session_state:
        st.download_button(
            "Download MP3 File",
            data=st.session_state["mp3_bytes"],
            file_name="generated_music.mp3",
            mime="audio/mpeg",
            key="dl_mp3",
        )

# ─── IN-APP PLAYBACK ─────────────────────────────────────────
if "pcm" in st.session_state:
    st.write("**Generated Melody:**")
    st.audio(
        st.session_state["pcm"],       # NumPy array
        sample_rate=16000              # now respected
    )
