import json

import config

RETRIEVED_FILE = config.OUTPUT_DIR / "retrieved.json"
RAG_FILE = config.OUTPUT_DIR / "results_rag.json"
MOTION_FILE = config.OUTPUT_DIR / "results_rag_motion.json"
REPORT_FILE = config.OUTPUT_DIR / "comparison.md"

# Short names for the 3 question types, in order.
SPLIT_NAMES = {
    "random_video_3d_what_move": "What is moving",
    "random_video_3d_where_move": "Where is it",
    "random_video_sg_what_object": "What object",
}


def load(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun the earlier steps first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def count_by_split(items, key):
    """Return {split: [number correct, number of questions]}."""
    counts = {split: [0, 0] for split in config.SPLITS}

    for item in items:
        counts[item["split"]][1] += 1
        counts[item["split"]][0] += int(bool(item[key]))

    return counts


def random_by_split(items):
    """Expected correct answers if we just guess."""
    counts = {split: [0.0, 0] for split in config.SPLITS}

    for item in items:
        counts[item["split"]][1] += 1
        counts[item["split"]][0] += 1 / len(item["options"])

    return counts


def format_cell(correct, total, is_float=False):
    if is_float:
        return f"{correct:.1f}/{total}"
    return f"{int(correct)}/{total}"


def make_row(name, counts, is_float=False):
    cells = [name]
    all_correct = 0
    all_total = 0

    for split in config.SPLITS:
        correct, total = counts[split]
        cells.append(format_cell(correct, total, is_float))
        all_correct += correct
        all_total += total

    percent = all_correct / all_total * 100
    cells.append(f"{format_cell(all_correct, all_total, is_float)} ({percent:.0f}%)")
    return cells


def main():
    retrieved = load(RETRIEVED_FILE)
    rag = load(RAG_FILE)
    motion = load(MOTION_FILE)

    rows = [
        make_row("Random guessing", random_by_split(retrieved), is_float=True),
        make_row("Search only (no LLM)", count_by_split(retrieved, "baseline_correct")),
        make_row("RAG + LLM", count_by_split(rag, "correct")),
        make_row("RAG + motion + LLM", count_by_split(motion, "correct")),
    ]

    header = ["Version"] + [SPLIT_NAMES[s] for s in config.SPLITS] + ["ALL"]

    # ---------- Per-question table ----------
    rag_by_id = {item["video_id"]: item for item in rag}
    motion_by_id = {item["video_id"]: item for item in motion}

    question_header = [
        "Video", "Type", "Right answer",
        "Search only", "RAG + LLM", "RAG + motion",
    ]
    question_rows = []

    for item in retrieved:
        video_id = item["video_id"]

        def mark(flag):
            return "yes" if flag else "no"

        question_rows.append([
            video_id,
            SPLIT_NAMES[item["split"]],
            str(item["answer"]),
            mark(item["baseline_correct"]),
            mark(rag_by_id[video_id]["correct"]),
            mark(motion_by_id[video_id]["correct"]),
        ])

    # ---------- Print ----------
    def print_table(head, body):
        widths = [
            max(len(str(row[i])) for row in [head] + body)
            for i in range(len(head))
        ]
        line = "  ".join(str(head[i]).ljust(widths[i]) for i in range(len(head)))
        print(line)
        print("-" * len(line))
        for row in body:
            print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))

    print("\nSCORE TABLE (correct / questions)\n")
    print_table(header, rows)

    print("\n\nQUESTION BY QUESTION\n")
    print_table(question_header, question_rows)

    # ---------- Markdown report ----------
    def md_table(head, body):
        lines = ["| " + " | ".join(head) + " |"]
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for row in body:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    report = f"""# Video question answering: results

Task: watch a short video (as pictures), answer a 4-option question.

## Score table (correct / questions)

{md_table(header, rows)}

## Question by question

{md_table(question_header, question_rows)}

## How each version works

- **Random guessing**: pick any option. About 25% right.
- **Search only (no LLM)**: write a note for each video frame, turn notes into numbers (embeddings), pick the option whose words are closest to the notes.
- **RAG + LLM**: same notes. Search finds the 5 notes closest to the question. A small language model reads them and picks an option.
- **RAG + motion + LLM**: same as above, plus a sentence from an OpenCV motion tracker (which way the object moves, where it sits in the frame).

## Things to remember

- Only 15 questions, and about 12 different videos (some videos are used by 2 or 3 questions). One question moves the score by about 7%. This is a good sign, not proof.
- The big gain is on the "Where is it" questions, because the motion tracker measures the answer directly.
- "What is moving" did not improve: the tracker says where and which way, but not what the object is.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\n\nSaved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
