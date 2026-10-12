from pathlib import Path

for version in ["original", "filtered"]:
    root = Path("dataset") / f"processed_{version}"
    print(f"\n{version.upper()}")

    for split in ["train", "val", "test"]:
        folder = root / split
        count = sum(1 for p in folder.rglob("*.png")) if folder.exists() else 0
        print(f"{split}: {count} images")