import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def pick_option(prediction, options):
    """
    Find which answer option appears in the model's text.
    - Several found: take the one that appears first.
    - Same start: take the longer one.
    - None found: return "" (counts as wrong).
    """
    pred = str(prediction).lower()
    best_option = ""
    best_key = None

    for option in options:
        opt = str(option).lower().strip()
        position = pred.find(opt)

        if position == -1:
            continue

        key = (position, -len(opt))

        if best_key is None or key < best_key:
            best_key = key
            best_option = opt

    return best_option


def build_prompt(item):
    frames_text = "\n".join(
        f"[t={frame['time_sec']}s] {frame['text']}"
        for frame in item["retrieved"]
    )

    options_text = "\n".join(f"- {option}" for option in item["options"])

    # Motion hint comes from step 7. Empty if there is none.
    motion_hint = item.get("motion_hint", "").strip()

    if motion_hint:
        motion_block = (
            "\nMotion measurement (from video tracking; reliable, "
            "trust it over the frame descriptions):\n"
            f"{motion_hint}\n"
        )
    else:
        motion_block = ""

    return f"""You are answering a question about a video.
You are given descriptions of the most relevant video frames, in time order.

Frame descriptions:
{frames_text}
{motion_block}
Question: {item["question"]}

Options:
{options_text}

Choose the most likely option, even if you are unsure.
Answer with exactly one option from the list above.
Write only the option text. Do not explain."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(config.OUTPUT_DIR / "retrieved.json"),
        help="JSON with retrieved captions per question",
    )
    parser.add_argument(
        "--output",
        default=str(config.OUTPUT_DIR / "results_rag.json"),
        help="Where to save scored results",
    )
    parser.add_argument(
        "--label",
        default="RAG (retrieved captions + LLM)",
        help="Name shown in the result table",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        items = json.load(file)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)

    print("Loading language model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.TEXT_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    split_stats = defaultdict(lambda: {"n": 0, "correct": 0})

    for item in items:
        prompt = build_prompt(item)

        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        answer = str(item["answer"]).lower().strip()
        chosen = pick_option(prediction, item["options"])
        correct = chosen == answer

        item["model_output"] = prediction
        item["chosen_option"] = chosen
        item["correct"] = correct

        stats = split_stats[item["split"]]
        stats["n"] += 1
        stats["correct"] += int(correct)

        print(f"\n{item['video_id']} | {item['question']}")
        if item.get("motion_hint"):
            print(f"  Motion hint: {item['motion_hint']}")
        print(f"  Model said: {prediction}")
        print(f"  Chosen: {chosen or '(none valid)'} | Answer: {answer} | {'OK' if correct else 'WRONG'}")

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(items, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"RESULT: {args.label}")

    total_n = 0
    total_correct = 0

    for split, stats in split_stats.items():
        accuracy = stats["correct"] / stats["n"] * 100
        print(f"  {split:30} {stats['correct']}/{stats['n']}  ({accuracy:.0f}%)")
        total_n += stats["n"]
        total_correct += stats["correct"]

    print(f"  {'ALL':30} {total_correct}/{total_n}  ({total_correct / total_n * 100:.0f}%)")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()