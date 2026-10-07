from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "datasets" / "dfire_raw"
SPLITS = ROOT / "datasets" / "dfire_splits"
FOLD = SPLITS / "cross_validation"

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}


def read_split(path: Path):
    names = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()

        # Handles accidental trailing "\" or "/" if present
        name = name.rstrip("\\/").strip()

        if name:
            names.append(Path(name).name)

    return names


def audit_split(split_name, filenames, source_dir):
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    missing_images = []
    missing_labels = []
    empty_labels = []
    invalid_labels = []

    class_counts = Counter()
    valid_boxes = 0

    for filename in filenames:
        image_path = images_dir / filename
        label_path = labels_dir / f"{Path(filename).stem}.txt"

        if not image_path.exists():
            missing_images.append(filename)
            continue

        if not label_path.exists():
            missing_labels.append(filename)
            continue

        content = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        # Empty label = background/negative image
        if not content:
            empty_labels.append(filename)
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            parts = line.split()

            if len(parts) != 5:
                invalid_labels.append(
                    f"{label_path.name}:{line_number} -> expected 5 values, got {len(parts)}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_labels.append(
                    f"{label_path.name}:{line_number} -> non-numeric value"
                )
                continue

            if class_id not in CLASS_NAMES:
                invalid_labels.append(
                    f"{label_path.name}:{line_number} -> invalid class {class_id}"
                )
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_labels.append(
                    f"{label_path.name}:{line_number} -> invalid YOLO coordinates"
                )
                continue

            class_counts[class_id] += 1
            valid_boxes += 1

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} AUDIT")
    print("=" * 60)

    print(f"Images listed       : {len(filenames)}")
    print(f"Missing images      : {len(missing_images)}")
    print(f"Missing label files : {len(missing_labels)}")
    print(f"Empty label files   : {len(empty_labels)}")
    print(f"Valid boxes         : {valid_boxes}")
    print(f"Smoke boxes (0)     : {class_counts[0]}")
    print(f"Fire boxes (1)      : {class_counts[1]}")
    print(f"Invalid annotations : {len(invalid_labels)}")

    if missing_images:
        print("\nFirst missing images:")
        for item in missing_images[:10]:
            print("  ", item)

    if missing_labels:
        print("\nFirst missing labels:")
        for item in missing_labels[:10]:
            print("  ", item)

    if invalid_labels:
        print("\nFirst invalid annotations:")
        for item in invalid_labels[:10]:
            print("  ", item)

    return {
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "invalid_labels": invalid_labels,
    }


# --------------------------------------------------
# Load official Fold 1
# --------------------------------------------------

train_files = read_split(FOLD / "dfire_train1.txt")
valid_files = read_split(FOLD / "dfire_valid1.txt")
test_files = read_split(SPLITS / "dfire_test.txt")


print("\nD-FIRE DATASET AUDIT")
print("=" * 60)

print(f"Train filenames      : {len(train_files)}")
print(f"Validation filenames : {len(valid_files)}")
print(f"Test filenames       : {len(test_files)}")


# --------------------------------------------------
# Check split overlap
# --------------------------------------------------

train_set = set(train_files)
valid_set = set(valid_files)
test_set = set(test_files)

train_valid_overlap = train_set & valid_set
train_test_overlap = train_set & test_set
valid_test_overlap = valid_set & test_set

print("\nSPLIT OVERLAP")
print("=" * 60)

print(f"Train ↔ Validation : {len(train_valid_overlap)}")
print(f"Train ↔ Test       : {len(train_test_overlap)}")
print(f"Validation ↔ Test  : {len(valid_test_overlap)}")


# --------------------------------------------------
# Audit each split
# --------------------------------------------------

train_result = audit_split(
    "Train",
    train_files,
    RAW / "train"
)

valid_result = audit_split(
    "Validation",
    valid_files,
    RAW / "train"
)

test_result = audit_split(
    "Test",
    test_files,
    RAW / "test"
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

critical_errors = (
    len(train_result["missing_images"])
    + len(valid_result["missing_images"])
    + len(test_result["missing_images"])
    + len(train_result["invalid_labels"])
    + len(valid_result["invalid_labels"])
    + len(test_result["invalid_labels"])
    + len(train_valid_overlap)
    + len(train_test_overlap)
    + len(valid_test_overlap)
)

print("\n" + "=" * 60)

if critical_errors == 0:
    print("✅ DATASET STRUCTURE AUDIT PASSED")
else:
    print("❌ DATASET AUDIT FOUND PROBLEMS")

print("=" * 60)

print("\nNOTE:")
print("Empty label files may legitimately represent negative/background images.")
print("Missing label files are reported separately and will be reviewed before training.")