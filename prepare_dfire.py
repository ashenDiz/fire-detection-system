from pathlib import Path
import shutil
from collections import Counter

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "datasets" / "dfire_raw"
SPLITS = ROOT / "datasets" / "dfire_splits"
CV = SPLITS / "cross_validation"

OUTPUT = ROOT / "datasets" / "dfire_yolo"

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}


def read_split(path: Path):
    names = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        name = line.strip().rstrip("\\/").strip()

        if name:
            names.append(Path(name).name)

    return names


def clean_label_file(label_path: Path):
    """
    Returns:
        cleaned_lines
        clipped_count
        dropped_count
        original_nonempty
    """

    raw_text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    # Legitimate negative/background image
    if not raw_text:
        return [], 0, 0, False

    cleaned = []
    clipped_count = 0
    dropped_count = 0

    for line in raw_text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            dropped_count += 1
            continue

        try:
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            dropped_count += 1
            continue

        if class_id not in CLASS_NAMES:
            dropped_count += 1
            continue

        # Completely unusable / degenerate box
        if w <= 0 or h <= 0:
            dropped_count += 1
            continue

        # Convert YOLO center coordinates to corners
        x1 = x - (w / 2)
        y1 = y - (h / 2)
        x2 = x + (w / 2)
        y2 = y + (h / 2)

        original = (x1, y1, x2, y2)

        # Clip to actual image boundaries
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Box vanished after clipping
        if x2 <= x1 or y2 <= y1:
            dropped_count += 1
            continue

        if original != (x1, y1, x2, y2):
            clipped_count += 1

        # Convert back to YOLO format
        new_x = (x1 + x2) / 2
        new_y = (y1 + y2) / 2
        new_w = x2 - x1
        new_h = y2 - y1

        cleaned.append(
            f"{class_id} "
            f"{new_x:.8f} "
            f"{new_y:.8f} "
            f"{new_w:.8f} "
            f"{new_h:.8f}"
        )

    return cleaned, clipped_count, dropped_count, True


def prepare_split(
    split_name,
    filenames,
    raw_source
):
    output_images = OUTPUT / split_name / "images"
    output_labels = OUTPUT / split_name / "labels"

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    source_images = raw_source / "images"
    source_labels = raw_source / "labels"

    copied = 0
    negative_images = 0
    excluded_images = 0
    clipped_boxes = 0
    dropped_boxes = 0

    class_counts = Counter()

    for filename in filenames:

        image_path = source_images / filename
        label_path = (
            source_labels /
            f"{Path(filename).stem}.txt"
        )

        if not image_path.exists():
            print(f"[MISSING IMAGE] {filename}")
            continue

        if not label_path.exists():
            print(f"[MISSING LABEL] {label_path.name}")
            continue

        (
            cleaned_lines,
            clipped,
            dropped,
            original_nonempty,
        ) = clean_label_file(label_path)

        clipped_boxes += clipped
        dropped_boxes += dropped

        # The original image had object labels but after cleaning
        # no valid object remains.
        # Exclude rather than falsely treating as background.
        if original_nonempty and not cleaned_lines:
            excluded_images += 1
            print(
                f"[EXCLUDED] {filename} "
                f"(all original boxes invalid)"
            )
            continue

        shutil.copy2(
            image_path,
            output_images / filename
        )

        cleaned_label_path = (
            output_labels /
            f"{Path(filename).stem}.txt"
        )

        if cleaned_lines:
            cleaned_label_path.write_text(
                "\n".join(cleaned_lines) + "\n",
                encoding="utf-8"
            )

            for line in cleaned_lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

        else:
            # Preserve legitimate D-Fire negative images
            cleaned_label_path.write_text(
                "",
                encoding="utf-8"
            )
            negative_images += 1

        copied += 1

    print()
    print("=" * 60)
    print(f"{split_name.upper()} PREPARATION")
    print("=" * 60)
    print(f"Images copied       : {copied}")
    print(f"Negative images     : {negative_images}")
    print(f"Excluded images     : {excluded_images}")
    print(f"Clipped boxes       : {clipped_boxes}")
    print(f"Dropped boxes       : {dropped_boxes}")
    print(f"Smoke boxes (0)     : {class_counts[0]}")
    print(f"Fire boxes (1)      : {class_counts[1]}")

    return {
        "copied": copied,
        "negative": negative_images,
        "excluded": excluded_images,
        "clipped": clipped_boxes,
        "dropped": dropped_boxes,
    }


def main():

    print()
    print("=" * 60)
    print("D-FIRE YOLO DATASET PREPARATION")
    print("=" * 60)

    if OUTPUT.exists():
        print()
        print(
            f"Removing previous generated dataset: {OUTPUT}"
        )
        shutil.rmtree(OUTPUT)

    train_files = read_split(
        CV / "dfire_train1.txt"
    )

    val_files = read_split(
        CV / "dfire_valid1.txt"
    )

    test_files = read_split(
        SPLITS / "dfire_test.txt"
    )

    train_result = prepare_split(
        "train",
        train_files,
        RAW / "train"
    )

    val_result = prepare_split(
        "val",
        val_files,
        RAW / "train"
    )

    test_result = prepare_split(
        "test",
        test_files,
        RAW / "test"
    )

    # Create YOLO configuration
    yaml_content = """path: .
train: train/images
val: val/images
test: test/images

names:
  0: smoke
  1: fire
"""

    (OUTPUT / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        f"Train images : {train_result['copied']}"
    )

    print(
        f"Val images   : {val_result['copied']}"
    )

    print(
        f"Test images  : {test_result['copied']}"
    )

    print(
        f"Total clipped boxes : "
        f"{train_result['clipped'] + val_result['clipped'] + test_result['clipped']}"
    )

    print(
        f"Total dropped boxes : "
        f"{train_result['dropped'] + val_result['dropped'] + test_result['dropped']}"
    )

    print(
        f"Total excluded images: "
        f"{train_result['excluded'] + val_result['excluded'] + test_result['excluded']}"
    )

    print()
    print("data.yaml created:")
    print(OUTPUT / "data.yaml")

    print()
    print("✅ D-Fire preparation completed.")
    print("Original dfire_raw dataset was NOT modified.")


if __name__ == "__main__":
    main()