import os

# -----------------------------
# Global research constants
# -----------------------------
RESEARCH_TITLE = "Fairness vs. Accuracy in CNNs"
EXPERIMENT_SHEET_ID = "1pA7K5EG36SCPi-jZEzb1wVFAP1S9TEVLZ0ogps2Bff0"

SEED = 42
MODEL_ARCH = "densenet121"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

CHEXPERT_ROOT = "/content/chexpert"
EVAL_INDEX_PATH = "/content/eval_reference/eval_index.csv"

DRIVE_ROOT = "/content/drive/MyDrive/Fairness-vs.-Accuracy-in-CNNs"
os.makedirs(DRIVE_ROOT, exist_ok=True)
RESULTS_PATH = f"{DRIVE_ROOT}/results_table.csv"

POS_LABEL = "PNEUMONIA"
NEG_LABEL = "NORMAL"

# Transform constants
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]
