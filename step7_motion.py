import json

import cv2
import numpy as np

import config

OUTPUT_FILE = config.OUTPUT_DIR / "motion.json"
DEBUG_DIR = config.OUTPUT_DIR / "motion_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

BLUR_SIZE = (5, 5)
DIFF_THRESHOLD = 25        # pixel change needed to count as motion
MIN_AREA_FRAC = 0.0005     # ignore tiny noise (fraction of frame area)
MIN_MOVE_FRAC = 0.03       # path shorter than 3% of frame = "no clear motion"
MIN_POINTS = 3             # need at least this many path points

# Where the front/back line sits (fraction of frame height, 0 = top).
# Lower part of the image = closer to camera = "front".
FRONT_LINE = 0.5


def read_video(path):
    """Return (blurred gray frames, first color frame, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    gray_frames = []
    first_color = None

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if first_color is None:
            first_color = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)
        gray_frames.append(gray)

    cap.release()
    return gray_frames, first_color, fps


def find_path(gray_frames):
    """Centre of changed pixels between each pair of neighbour frames."""
    height, width = gray_frames[0].shape
    points = []

    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i - 1], gray_frames[i])
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Remove tiny specks.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        area = cv2.countNonZero(mask)

        if area < MIN_AREA_FRAC * height * width:
            continue

        moments = cv2.moments(mask, binaryImage=True)

        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        points.append((i, cx, cy))

    return points, width, height


def analyze_path(points, width, height, n_frames):
    """Turn the path into direction + position words."""
    if len(points) < MIN_POINTS:
        return None

    t = np.array([p[0] for p in points], dtype=float)
    x = np.array([p[1] for p in points], dtype=float)
    y = np.array([p[2] for p in points], dtype=float)

    # Straight-line fit: how far did it travel over the whole video?
    slope_x = np.polyfit(t, x, 1)[0]
    slope_y = np.polyfit(t, y, 1)[0]

    dx = slope_x * (n_frames - 1)
    dy = slope_y * (n_frames - 1)   # image y grows downward, so dy>0 = down

    if max(abs(dx) / width, abs(dy) / height) < MIN_MOVE_FRAC:
        direction = "no clear movement"
    elif abs(dy) / height >= abs(dx) / width:
        direction = "down" if dy > 0 else "up"
    else:
        direction = "right" if dx > 0 else "left"

    mean_x = float(x.mean()) / width
    mean_y = float(y.mean()) / height

    side = "right" if mean_x >= 0.5 else "left"
    depth = "front" if mean_y >= FRONT_LINE else "back"

    return {
        "points": len(points),
        "dx_px": round(float(dx), 1),
        "dy_px": round(float(dy), 1),
        "direction": direction,
        "mean_x": round(mean_x, 3),
        "mean_y": round(mean_y, 3),
        "position": f"{depth} {side}",
    }


def make_hint(result):
    """One short sentence for the answer model."""
    if result is None:
        return "No reliable movement was measured."

    return (
        f"Measured by motion tracking: the moving object travels mainly "
        f"{result['direction']}. It is located in the {result['position']} "
        f"part of the frame."
    )


def save_debug_image(first_color, points, result, path):
    """Draw the path on the first frame, so you can check by eye."""
    image = first_color.copy()

    for index in range(1, len(points)):
        a = (int(points[index - 1][1]), int(points[index - 1][2]))
        b = (int(points[index][1]), int(points[index][2]))
        cv2.line(image, a, b, (0, 255, 0), 2)

    for _, cx, cy in points:
        cv2.circle(image, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    label = "no motion" if result is None else f"{result['direction']} | {result['position']}"
    cv2.putText(
        image, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    cv2.imwrite(str(path), image)


def main():
    with open(config.SAMPLES_FILE, "r", encoding="utf-8") as file:
        samples = json.load(file)

    motion = {}

    for sample in samples:
        video_id = sample["video_id"]
        video_path = config.ROOT / sample["video_path"]

        gray_frames, first_color, fps = read_video(video_path)

        if len(gray_frames) < 2 or fps <= 0:
            print(f"{video_id}: could not read video.")
            continue

        duration = len(gray_frames) / fps

        # Long videos (people on a bench) are not motion clips. Skip.
        if duration > config.SHORT_VIDEO_MAX_SEC:
            print(f"{video_id}: {duration:.0f}s long video. Skipped.")
            continue

        points, width, height = find_path(gray_frames)
        result = analyze_path(points, width, height, len(gray_frames))

        motion[video_id] = {
            "result": result,
            "hint": make_hint(result),
        }

        save_debug_image(
            first_color, points, result, DEBUG_DIR / f"{video_id}.jpg"
        )

        print(f"\n{video_id} | {sample['question']}")
        print(f"  Answer: {sample['answer']}")

        if result is None:
            print("  Measured: no reliable motion")
        else:
            print(
                f"  Measured: direction={result['direction']} "
                f"position={result['position']} "
                f"(x={result['mean_x']}, y={result['mean_y']}, "
                f"dx={result['dx_px']}px, dy={result['dy_px']}px)"
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(motion, file, indent=4, ensure_ascii=False)

    # ---------- Check against known answers ----------
    print("\n" + "=" * 50)
    print("MOTION TOOL CHECK")

    where_ok = where_n = 0
    down_ok = down_n = 0

    for sample in samples:
        video_id = sample["video_id"]
        entry = motion.get(video_id)

        if entry is None or entry["result"] is None:
            if sample["split"] == "random_video_3d_where_move":
                where_n += 1
            continue

        result = entry["result"]
        question = sample["question"].lower()

        if sample["split"] == "random_video_3d_where_move":
            where_n += 1
            where_ok += int(result["position"] == str(sample["answer"]).lower())

        if "moving down" in question:
            down_n += 1
            down_ok += int(result["direction"] == "down")

    print(f"  'Where' position correct:   {where_ok}/{where_n}")
    print(f"  'Moving down' direction ok: {down_ok}/{down_n}")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Debug pictures: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
