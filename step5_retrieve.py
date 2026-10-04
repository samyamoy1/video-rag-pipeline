import json
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

VECTORS_FILE = config.OUTPUT_DIR / "index_vectors.npy"
RECORDS_FILE = config.OUTPUT_DIR / "index_records.json"
OUTPUT_FILE = config.OUTPUT_DIR / "retrieved.json"

TOP_K = 5


def main():
    # ---------- Load ----------
    with open(config.SAMPLES_FILE, "r", encoding="utf-8") as file:
        samples = json.load(file)

    with open(RECORDS_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    vectors = np.load(VECTORS_FILE)

    # Row numbers of each video inside the index.
    video_rows = defaultdict(list)
    for row, record in enumerate(records):
        video_rows[record["video_id"]].append(row)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)

    results = []
    split_stats = defaultdict(lambda: {"n": 0, "baseline": 0, "in_top": 0, "in_video": 0})

    for sample in samples:
        video_id = sample["video_id"]
        question = sample["question"]
        options = sample["options"]
        answer = str(sample["answer"]).lower().strip()

        rows = video_rows.get(video_id, [])

        if not rows:
            print(f"{video_id}: no captions in index. Skipping.")
            continue

        video_vectors = vectors[rows]

        # ---------- 1. Retrieve top-k captions ----------
        query_text = f"{question} {', '.join(options)}"
        query_vector = model.encode(
            [config.QUERY_PREFIX + query_text],
            normalize_embeddings=True,
        )[0]

        scores = video_vectors @ query_vector
        top_positions = np.argsort(-scores)[:TOP_K]

        retrieved = []
        for position in top_positions:
            record = records[rows[position]]
            retrieved.append({
                "frame": record["frame"],
                "time_sec": record["time_sec"],
                "score": round(float(scores[position]), 4),
                "text": record["text"],
            })

        # Show in time order (the answer model reads it like a story).
        retrieved.sort(key=lambda item: item["time_sec"])

        # ---------- 2. Option similarity baseline ----------
        option_vectors = model.encode(
            [config.QUERY_PREFIX + str(option) for option in options],
            normalize_embeddings=True,
        )

        # For each option: best match against any caption of this video.
        option_scores = {}
        for option, option_vector in zip(options, option_vectors):
            option_scores[str(option)] = round(float((video_vectors @ option_vector).max()), 4)

        baseline_pred = max(option_scores, key=option_scores.get).lower()

        # ---------- 3. Diagnostics ----------
        retrieved_text = " ".join(item["text"].lower() for item in retrieved)
        all_text = " ".join(records[row]["text"].lower() for row in rows)

        answer_in_top = answer in retrieved_text
        answer_in_video = answer in all_text

        results.append({
            "video_id": video_id,
            "split": sample["split"],
            "question": question,
            "options": options,
            "answer": sample["answer"],
            "retrieved": retrieved,
            "option_scores": option_scores,
            "baseline_pred": baseline_pred,
            "baseline_correct": baseline_pred == answer,
            "answer_in_top_k": answer_in_top,
            "answer_in_any_caption": answer_in_video,
        })

        stats = split_stats[sample["split"]]
        stats["n"] += 1
        stats["baseline"] += int(baseline_pred == answer)
        stats["in_top"] += int(answer_in_top)
        stats["in_video"] += int(answer_in_video)

        print(f"\n{video_id} | {question}")
        print(f"  Answer: {answer} | Baseline pick: {baseline_pred}")
        print(f"  Answer word in top-{TOP_K}: {answer_in_top} | in any caption: {answer_in_video}")
        print(f"  Option scores: {option_scores}")

    # ---------- Save ----------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("SUMMARY (per split)")
    print(f"{'split':32} {'n':>3} {'baseline':>9} {'in top-k':>9} {'in video':>9}")

    total = {"n": 0, "baseline": 0, "in_top": 0, "in_video": 0}

    for split, stats in split_stats.items():
        print(
            f"{split:32} {stats['n']:>3} "
            f"{stats['baseline']:>9} {stats['in_top']:>9} {stats['in_video']:>9}"
        )
        for key in total:
            total[key] += stats[key]

    print(
        f"{'ALL':32} {total['n']:>3} "
        f"{total['baseline']:>9} {total['in_top']:>9} {total['in_video']:>9}"
    )
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
