import os
from pathlib import Path

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
RANDOM_STATE = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLITS = ('train', 'val', 'test')
IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

ROOT = Path(os.environ.get('SEM_PROJECT_DIR', Path(__file__).resolve().parent))
RAW_DIR = ROOT / 'dataset' / 'raw'
MANIFEST = ROOT / 'dataset' / 'manifest.csv'
PROCESSED = {'original': ROOT / 'dataset' / 'processed_original',
             'filtered': ROOT / 'dataset' / 'processed_filtered'}
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
