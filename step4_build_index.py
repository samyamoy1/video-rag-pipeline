import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

CAPTIONS_FILE = config.OUTPUT_DIR / "captions.json"
VECTORS_FILE = config.OUTPUT_DIR / "index_vectors.npy"
RECORDS_FILE = config.OUTPUT_DIR / "index_records.json"


def clean_caption(text):
    """Remove markdown bullets and bold marks so text is plain."""
    text = text.replace("**", "")
    text = text.replace("\n", " ")
    text = text.replace(" - ", " ")
    text = text.lstrip("- ").strip()
    return " ".join(text.split())


def main():
    # ---------- Load captions ----------
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:
        captions = json.load(file)

    records = []

    for video_id, items in captions.items():
        for item in items:
            records.append({
                "video_id": video_id,
                "frame": item["frame"],
                "time_sec": item["time_sec"],
                "caption": item["caption"],
                "text": clean_caption(item["caption"]),
            })

    print(f"Captions to embed: {len(records)}")

    # ---------- Embed ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer(config.EMBED_MODEL, device=device)

    texts = [record["text"] for record in records]

    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # unit length -> dot product = cosine
        show_progress_bar=True,
    )

    print("Vector shape:", vectors.shape)

    # ---------- Save ----------
    np.save(VECTORS_FILE, vectors)

    with open(RECORDS_FILE, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4, ensure_ascii=False)

    print(f"Saved vectors: {VECTORS_FILE}")
    print(f"Saved records: {RECORDS_FILE}")

    # ---------- Quick search test ----------
    print("\n" + "=" * 50)
    print("SEARCH TEST (video v3_0 only)")

    query = "a person covered with a red blanket"
    print("Query:", query)

    query_vector = model.encode(
        [config.QUERY_PREFIX + query],
        normalize_embeddings=True,
    )[0]

    # Only look inside one video, like a real question will.
    indices = [i for i, r in enumerate(records) if r["video_id"] == "v3_0"]
    scores = vectors[indices] @ query_vector

    top = np.argsort(-scores)[:3]

    for rank, position in enumerate(top, start=1):
        record = records[indices[position]]
        print(
            f"\n#{rank} score={scores[position]:.3f} "
            f"{record['frame']} (t={record['time_sec']}s)"
        )
        print("   ", record["text"][:140])


if __name__ == "__main__":
    main()