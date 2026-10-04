import json

from datasets import load_dataset

import config


def main():
    all_samples = []

    for split_index, split_name in enumerate(config.SPLITS, start=1):
        print(f"\nLoading split: {split_name}")

        dataset = load_dataset(config.DATASET_NAME, split=split_name)

        count = min(config.SAMPLES_PER_SPLIT, len(dataset))
        print(f"Split has {len(dataset)} samples. Taking {count}.")

        for i in range(count):
            sample = dataset[i]

            # Unique name per video: v1_0, v1_1, ... v2_0, ...
            video_id = f"v{split_index}_{i}"
            video_path = config.VIDEO_DIR / f"{video_id}.mp4"
            video_path.write_bytes(sample["video"])

            all_samples.append({
                "video_id": video_id,
                "video_path": video_path.relative_to(config.ROOT).as_posix(),
                "split": split_name,
                "question": sample["question"],
                "options": sample["options"],
                "answer": sample["answer"],
            })

            print(f"  Saved {video_id}: {sample['question']}")
            print(f"    Options: {sample['options']}")
            print(f"    Answer:  {sample['answer']}")

    with open(config.SAMPLES_FILE, "w", encoding="utf-8") as file:
        json.dump(all_samples, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Total samples saved: {len(all_samples)}")
    print(f"Metadata file: {config.SAMPLES_FILE}")
    print(f"Videos folder: {config.VIDEO_DIR}")


if __name__ == "__main__":
    main()
