"""
Register Chihuahua vs Muffin dataset in 3LC tables.

Creates 3LC tables for train and val only. The test set (data/test/) is never
registered or used for training; it is only read by predict.py for inference.

Idempotency: Safe to run multiple times. If train and val tables already exist
for this project/dataset, the script skips registration and does not overwrite
them. To recreate tables from disk, delete them in the 3LC Dashboard first, then
run this script again.

Expected folder structure:
    data/
    ├── train/
    │   ├── chihuahua/    (labeled)
    │   ├── muffin/       (labeled)
    │   └── undefined/    (unlabeled - your opportunity!)
    ├── val/
    │   ├── chihuahua/
    │   └── muffin/
    └── test/             (FLAT - just images, no subfolders, for Kaggle submission)

Classes: 0 = chihuahua, 1 = muffin, 2 = undefined (train only)
"""

import tlc
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSES = ["chihuahua", "muffin", "undefined"]
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

schemas = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=CLASSES),
    "weight": tlc.SampleWeightSchema(),
}


def register_dataset_to_table(
    dataset_path: Path,
    table_name: str,
    split_name: str,
    include_undefined: bool = False,
):
    """
    Register images from folder structure to a 3LC table.
    Each folder (chihuahua, muffin, undefined) corresponds to a class.
    """
    dataset_path = Path(dataset_path)
    image_data = []

    classes_to_process = CLASSES[:-1] if not include_undefined else CLASSES

    for class_idx, class_name in enumerate(classes_to_process):
        class_folder = dataset_path / class_name
        if class_folder.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_files = sorted(class_folder.glob(ext))
                if image_files:
                    print(f"  Found {len(image_files):5d} images in {class_name}/ for {split_name}")
                for img_path in image_files:
                    label = class_idx if class_name != "undefined" else 2
                    image_data.append({"path": str(img_path.absolute()), "label": label})
        else:
            if class_name != "undefined":
                print(f"  [WARN] {class_folder} does not exist")

    print(f"\n  Total images for {split_name}: {len(image_data)}")

    # if_exists="overwrite" only applies when we actually create (first run).
    # When tables already exist, main() returns early and this is never called.
    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Chihuahua vs Muffin {split_name} set with {len(image_data)} images",
        column_schemas=schemas,
        if_exists="overwrite",
    )

    for i, data in enumerate(image_data):
        label = data["label"]
        weight = 1.0 if label in [0, 1] else 0.0
        table_writer.add_row({
            "id": i,
            "image": data["path"],
            "label": label,
            "weight": weight,
        })

    table = table_writer.finalize()
    num_labeled = sum(1 for x in image_data if x["label"] in [0, 1])
    num_undefined = sum(1 for x in image_data if x["label"] == 2)
    print(f"\n  [OK] Created 3LC table: '{table_name}'")
    print(f"  Labeled (weight=1.0): {num_labeled}, Undefined (weight=0.0): {num_undefined}")
    print(f"  Table URL: {table.url}")
    return table


def tables_exist():
    """
    Return True if both train and val tables already exist for this project/dataset.
    Used for idempotency: when True, we skip registration and do not overwrite.
    """
    try:
        train_ref = tlc.Table.from_names(
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            table_name="train",
        )
        val_ref = tlc.Table.from_names(
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            table_name="val",
        )
        return True, train_ref, val_ref
    except Exception:
        return False, None, None


def main():
    base_path = Path(__file__).parent
    data_path = base_path / "data"

    print("=" * 70)
    print("  Registering Chihuahua vs Muffin Dataset in 3LC Tables")
    print("=" * 70)

    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found: {data_path}")
        print("  Create data/ and add train/, val/, test/ (see README and dataset description).")
        return

    exist, train_ref, val_ref = tables_exist()
    if exist:
        print("\n  [IDEMPOTENT] Train and val tables already exist. Skipping registration.")
        print("  No tables were overwritten. Safe to run this script multiple times.")
        if train_ref is not None and val_ref is not None:
            try:
                train_table = train_ref.latest()
                val_table = val_ref.latest()
                print(f"  Train table URL: {train_table.url}")
                print(f"  Val table URL:   {val_table.url}")
            except Exception:
                pass
        print("  To recreate from disk: delete the tables in 3LC Dashboard, then run again.")
        print("=" * 70 + "\n")
        return

    print("\nRegistering URL alias for portable paths...")
    tlc.register_project_url_alias(
        token="CHIHUAHUA_MUFFIN_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )
    print(f"  [OK] Alias registered -> {base_path.absolute()}")

    print("\n" + "-" * 70)
    print("[1/2] Registering TRAIN set (includes undefined)...")
    print("-" * 70)
    train_table = register_dataset_to_table(
        data_path / "train",
        table_name="train",
        split_name="train",
        include_undefined=True,
    )

    print("\n" + "-" * 70)
    print("[2/2] Registering VAL set...")
    print("-" * 70)
    val_table = register_dataset_to_table(
        data_path / "val",
        table_name="val",
        split_name="val",
        include_undefined=False,
    )

    print("\n" + "=" * 70)
    print("  [OK] Successfully registered tables!")
    print("  Test set is FLAT - use predict.py to generate submission.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
