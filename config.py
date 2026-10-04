from pathlib import Path

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAMES_DIR = ROOT / "frames"
OUTPUT_DIR = ROOT / "outputs"

# ---------- Files ----------
SAMPLES_FILE = DATA_DIR / "samples.json"

# ---------- Dataset ----------
DATASET_NAME = "weikaih/TaskMeAnything-v1-videoqa-random"

SPLITS = [
    "random_video_3d_what_move",
    "random_video_3d_where_move",
    "random_video_sg_what_object",
]

# How many videos to take from each split.
SAMPLES_PER_SPLIT = 5

# ---------- Models ----------
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge models work best when search queries get this prefix.
# (Captions/documents get NO prefix.)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------- Frames (step 2) ----------
SHORT_VIDEO_MAX_SEC = 10         # video at or below this = "short"
FRAME_INTERVAL_SHORT_SEC = 0.25  # short video: 4 frames per second
FRAME_INTERVAL_LONG_SEC = 1.0    # long video: 1 frame per second

# ---------- Captioning speed (step 3) ----------
# True = load vision model in 4-bit (fits 6 GB GPU, much faster).
VLM_LOAD_4BIT = True

# Caption every Nth frame. 1 = all frames, 2 = every other frame.
CAPTION_EVERY_N_SHORT = 2
CAPTION_EVERY_N_LONG = 2

# Smaller image = faster. 256 is fast, 512 is more detailed.
CAPTION_MAX_PIXELS = 256 * 28 * 28

# Shorter caption = faster.
CAPTION_MAX_NEW_TOKENS = 70

# ---------- Create folders ----------
for folder in (DATA_DIR, VIDEO_DIR, FRAMES_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)