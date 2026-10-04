import json

import config

RETRIEVED_FILE = config.OUTPUT_DIR / "retrieved.json"
MOTION_FILE = config.OUTPUT_DIR / "motion.json"
OUTPUT_FILE = config.OUTPUT_DIR / "retrieved_motion.json"


def main():
    with open(RETRIEVED_FILE, "r", encoding="utf-8") as file:
        items = json.load(file)

    with open(MOTION_FILE, "r", encoding="utf-8") as file:
        motion = json.load(file)

    with_hint = 0

    for item in items:
        entry = motion.get(item["video_id"])

        # Only add a hint when motion was really measured.
        if entry and entry["result"] is not None:
            item["motion_hint"] = entry["hint"]
            with_hint += 1
        else:
            item["motion_hint"] = ""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(items, file, indent=4, ensure_ascii=False)

    print(f"Questions: {len(items)}")
    print(f"With motion hint: {with_hint}")
    print(f"Without (kept as before): {len(items) - with_hint}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
