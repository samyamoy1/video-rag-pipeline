import json
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

import config

OUTPUT_FILE = config.OUTPUT_DIR / "captions.json"

PROMPT = (
    "Describe only what is visible in this frame. "
    "List the main objects, their colors, and where each one is "
    "(left or right, front or back, top or bottom of the frame). "
    "Do not guess uncertain details. "
    "Use no more than 40 words."
)


def frame_number(frame_name):
    """frame_12.jpg -> 12 (true time order)."""
    return int(frame_name.split("_")[-1].split(".")[0])


def save_captions(captions):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(captions, file, indent=4, ensure_ascii=False)


def select_frames(frames_info):
    """Keep every Nth frame, depending on short or long video."""
    last_time = frames_info[-1]["time_sec"]
    is_short = last_time <= config.SHORT_VIDEO_MAX_SEC

    step = config.CAPTION_EVERY_N_SHORT if is_short else config.CAPTION_EVERY_N_LONG
    selected = frames_info[::step]

    # Always keep the last frame, so the end of the video is covered.
    if selected[-1] != frames_info[-1]:
        selected.append(frames_info[-1])

    return selected


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")

    with open(config.SAMPLES_FILE, "r", encoding="utf-8") as file:
        samples = json.load(file)

    # Resume from earlier progress.
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            captions = json.load(file)
    else:
        captions = {}

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        config.VLM_MODEL,
        min_pixels=64 * 28 * 28,
        max_pixels=config.CAPTION_MAX_PIXELS,
    )

    print("Loading Qwen vision model...")

    if device == "cuda" and config.VLM_LOAD_4BIT:
        print("Mode: 4-bit (fits fully on GPU)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL,
            quantization_config=quant_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        print("Mode: full precision")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model.eval()

    if device == "cuda":
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used by model: {used:.1f} GB")

    # Build the full to-do list first, so we can show progress.
    todo = []

    for sample in samples:
        video_id = sample["video_id"]
        frame_dir = config.FRAMES_DIR / video_id
        info_file = frame_dir / "frames_info.json"

        if not info_file.exists():
            print(f"{video_id}: no frames_info.json. Run step 2 first.")
            continue

        with open(info_file, "r", encoding="utf-8") as file:
            frames_info = json.load(file)

        frames_info.sort(key=lambda item: frame_number(item["frame"]))
        captions.setdefault(video_id, [])
        done = {item["frame"] for item in captions[video_id]}

        for info in select_frames(frames_info):
            if info["frame"] not in done:
                todo.append((video_id, frame_dir, info))

    total = len(todo)
    print(f"\nFrames left to caption: {total}")

    start = time.time()

    for count, (video_id, frame_dir, info) in enumerate(todo, start=1):
        frame_name = info["frame"]

        with Image.open(frame_dir / frame_name) as image_file:
            image = image_file.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config.CAPTION_MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated = output[:, inputs["input_ids"].shape[1]:]
        caption = processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

        captions[video_id].append({
            "frame": frame_name,
            "time_sec": info["time_sec"],
            "caption": caption,
        })

        # Keep each video's captions in time order.
        captions[video_id].sort(key=lambda item: frame_number(item["frame"]))
        save_captions(captions)

        elapsed = time.time() - start
        per_frame = elapsed / count
        remaining_min = per_frame * (total - count) / 60

        print(
            f"[{count}/{total}] {video_id} {frame_name} "
            f"({per_frame:.1f}s/frame, ~{remaining_min:.0f} min left)"
        )
        print(f"    {caption[:100]}")

        del inputs, output, generated
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nFinished. Captions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()