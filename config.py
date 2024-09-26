import os

# Random seed for reproducibility
SEED = 7

# Base directory
BASE_DIR = "/path/to/WildFusion"

# Paths
SAVE_DIR = os.path.join(BASE_DIR, "saved_model_test_full")
TRAIN_FILE_PATH = os.path.join(BASE_DIR, "input/train_data.npz")
VAL_FILE_PATH = os.path.join(BASE_DIR, "input/val_data.npz")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# Training parameters
BATCH_SIZE = 8
EPOCHS = 300
LR = 0.00002
EARLY_STOP_EPOCHS = 25
PLOT_INTERVAL = 3

# Loss weights
WEIGHT_SDF = 20.0
WEIGHT_CONFIDENCE = 1.0
WEIGHT_SEMANTICS = 1.0
WEIGHT_COLOR = 1.0
WEIGHT_TRAV = 1.0
WEIGHT_EL = 0.05

# Other parameters
NUM_BINS = 313
POINTS_PER_SCAN = 30000
CLASSES = 11
PATIENCE = 5
LR_DECAY_FACTOR = 0.6