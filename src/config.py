from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"

METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIR_1 = DATA_DIR / "HAM10000_images_part_1"
IMAGE_DIR_2 = DATA_DIR / "HAM10000_images_part_2"
SPLIT_DIR = DATA_DIR / "splits"
CHECKPOINT_PATH = MODEL_DIR / "densenet121_ham10000.pt"

IMAGE_SIZE = 160
NUM_CLASSES = 7
CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4

FREEZE_BACKBONE = True


# The public model is intended for demonstration and research only.
APP_TITLE = "Explainable Skin Lesion Classification"
