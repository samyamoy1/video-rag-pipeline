import json

import cv2

import config


def extract_video_frames(video_path, output_dir):
    """
    Save frames at an interval chosen by video length.
    Return list of {"frame": filename, "time_sec": seconds}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  Could not open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        print(f"  Bad video info (fps={fps}, frames={total_frames})")
        cap.release()
        return []

    duration = total_frames / fps

    if duration <= config.SHORT_VIDEO_MAX_SEC:
        interval = config.FRAME_INTERVAL_SHORT_SEC
    else:
        interval = config.FRAME_INTERVAL_LONG_SEC

    print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, interval: {interval}s")

    frames_info = []
    second = 0.0
    index = 0

    while second < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = cap.read()

        if not success:
            break

        frame_name = f"frame_{index}.jpg"
        cv2.imwrite(str(output_dir / frame_name), frame)

        frames_info.append({
            "frame": frame_name,
            "time_sec": round(second, 2),
        })

        index += 1
        second += interval

    cap.release()

    # Save frame -> time map, so later steps know real timestamps.
    with open(output_dir / "frames_info.json", "w", encoding="utf-8") as file:
        json.dump(frames_info, file, indent=4)

    return frames_info


def main():
    with open(config.SAMPLES_FILE, "r", encoding="utf-8") as file:
        samples = json.load(file)

    print(f"Videos to process: {len(samples)}")

    for sample in samples:
        video_id = sample["video_id"]
        video_path = config.ROOT / sample["video_path"]
        output_dir = config.FRAMES_DIR / video_id

        print(f"\n{video_id}")

        # Skip videos already done, so you can rerun safely.
        if (output_dir / "frames_info.json").exists():
            print("  Already done. Skipping.")
            continue

        frames_info = extract_video_frames(video_path, output_dir)
        print(f"  Saved {len(frames_info)} frames -> {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
