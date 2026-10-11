import hashlib
import random
from pathlib import Path
import pandas as pd
from PIL import Image
import config as C

csvs = sorted(C.RAW_DIR.rglob('*.csv'))
assert len(
    csvs) == 1, f'Expected exactly one CSV under {C.RAW_DIR}, found: {csvs}'
labels = labels = pd.read_csv(csvs[0], sep=';')
print('CSV file:', csvs[0])
print('Columns :', list(labels.columns))
print(labels.head(), '\n')

# Column names
NAME_COL = 'file_name'
PATH_COL = 'image_path'
CLASS_COL = 'label'
for col in (NAME_COL, PATH_COL, CLASS_COL):
    assert col in labels.columns, f'Column {col!r} not found. Columns are: {list(labels.columns)}'

# Index the image files that are actually on disk
on_disk = [p for p in C.RAW_DIR.rglob(
    '*') if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS]
by_name = {}
for p in on_disk:
    by_name.setdefault(p.name, []).append(p)
clashes = {k: v for k, v in by_name.items() if len(v) > 1}
assert not clashes, f'Same file name in several folders: {list(clashes)[:3]}'
by_stem = {Path(n).stem: v[0] for n, v in by_name.items()}
print(f'{len(on_disk)} image files on disk, {len(labels)} rows in the CSV')

# Match every CSV row to a file, verify it, drop exact duplicates
rows, corrupt, missing, seen = [], [], [], set()
for _, r in labels.iterrows():
    name = Path(str(r[NAME_COL])).name
    p = by_name.get(name, [None])[0] or by_stem.get(Path(name).stem)
    if p is None:
        missing.append(name)
        continue
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
    except Exception as e:
        corrupt.append({'filepath': p.name, 'error': str(e)})
        continue
    digest = hashlib.md5(p.read_bytes()).hexdigest()
    if digest in seen:            # exact duplicate -> would leak between splits
        continue
    seen.add(digest)
    rows.append({'filepath': p.relative_to(C.RAW_DIR).as_posix(),
                 'class': str(r[CLASS_COL]), 'width': w, 'height': h})

df = pd.DataFrame(rows)
used = set(df['filepath'])
unlisted = [p.name for p in on_disk if p.relative_to(
    C.RAW_DIR).as_posix() not in used]
print(f'{len(df)} usable | {len(missing)} CSV rows with no image file | '
      f'{len(corrupt)} corrupt | {len(unlisted)} images not used (not in CSV, corrupt or duplicate)')

# Stratified per-class split (70/15/15)
rng = random.Random(C.RANDOM_STATE)
df['split'] = ''
for cls, grp in df.groupby('class'):
    idx = grp.index.tolist()
    assert len(idx) >= 3, f'class {cls} is too small to split'
    rng.shuffle(idx)
    n_test = max(1, round(len(idx) * C.TEST_FRAC))
    n_val = max(1, round(len(idx) * C.VAL_FRAC))
    df.loc[idx[:n_test], 'split'] = 'test'
    df.loc[idx[n_test:n_test + n_val], 'split'] = 'val'
    df.loc[idx[n_test + n_val:], 'split'] = 'train'

C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(C.MANIFEST, index=False)
pd.DataFrame(corrupt, columns=['filepath', 'error']).to_csv(
    C.RESULTS_DIR / 'corrupt_files.csv', index=False)
pd.DataFrame({'csv_name_without_image': missing}).to_csv(
    C.RESULTS_DIR / 'missing_images.csv', index=False)
print(pd.crosstab(df['class'], df['split']))
