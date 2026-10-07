from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "datasets" / "dfire_raw"
SPLITS = ROOT / "datasets" / "dfire_splits"
FOLD = SPLITS / "cross_validation"

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}


def read_split(path):
    names = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        name = line.strip().rstrip("\\/").strip()

        if name:
            names.append(Path(name).name)

    return names


def inspect(split_name, filenames, source_dir):

    labels_dir = source_dir / "labels"

    print()
    print("=" * 80)
    print(split_name.upper())
    print("=" * 80)

    problem_count = 0

    for filename in filenames:

        label_path = labels_dir / f"{Path(filename).stem}.txt"

        if not label_path.exists():
            continue

        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"{label_path.name}:{line_number}"
                    f" | INVALID FIELD COUNT"
                    f" | {line}"
                )
                problem_count += 1
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                print(
                    f"{label_path.name}:{line_number}"
                    f" | NON-NUMERIC"
                    f" | {line}"
                )
                problem_count += 1
                continue

            reasons = []

            if class_id not in CLASS_NAMES:
                reasons.append(
                    f"invalid class={class_id}"
                )

            if not 0 <= x <= 1:
                reasons.append(f"x={x}")

            if not 0 <= y <= 1:
                reasons.append(f"y={y}")

            if not 0 < w <= 1:
                reasons.append(f"w={w}")

            if not 0 < h <= 1:
                reasons.append(f"h={h}")

            # Also check whether the box crosses image boundaries
            x1 = x - w / 2
            x2 = x + w / 2
            y1 = y - h / 2
            y2 = y + h / 2

            if x1 < 0:
                reasons.append(f"left={x1:.6f}")

            if x2 > 1:
                reasons.append(f"right={x2:.6f}")

            if y1 < 0:
                reasons.append(f"top={y1:.6f}")

            if y2 > 1:
                reasons.append(f"bottom={y2:.6f}")

            if reasons:
                problem_count += 1

                class_name = CLASS_NAMES.get(
                    class_id,
                    "UNKNOWN"
                )

                print(
                    f"{label_path.name}:{line_number}"
                )

                print(
                    f"  class : {class_id} ({class_name})"
                )

                print(
                    f"  raw   : {line}"
                )

                print(
                    f"  issue : {', '.join(reasons)}"
                )

                print()

    print(
        f"{split_name} problem boxes: {problem_count}"
    )

    return problem_count


train = read_split(
    FOLD / "dfire_train1.txt"
)

valid = read_split(
    FOLD / "dfire_valid1.txt"
)

test = read_split(
    SPLITS / "dfire_test.txt"
)


total = 0

total += inspect(
    "Train",
    train,
    RAW / "train"
)

total += inspect(
    "Validation",
    valid,
    RAW / "train"
)

total += inspect(
    "Test",
    test,
    RAW / "test"
)

print()
print("=" * 80)
print(f"TOTAL PROBLEM BOXES: {total}")
print("=" * 80)