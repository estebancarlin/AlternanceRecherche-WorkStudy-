# -------------------------------
# General Data Parameters
# -------------------------------
DATA_PATH = './data'
TRAINING_DATA_PATH = DATA_PATH
TESTING_DATA_PATH = DATA_PATH
SHUFFLE_DATA = True
COMMON_SIZE = (216, 256, 9)
DATA_AUGMENTATION = True
TRANSFORM_DATA = None  # Define transformations if needed

# Data split ratios
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TESTING_SIZE = 0.2

# -------------------------------
# Data Generator Parameters
# -------------------------------
TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
TESTING_BATCH_SIZE = 16

# -------------------------------
# Model Parameters
# -------------------------------
DROPOUT_RATE = 0.0

# -------------------------------
# Training Parameters
# -------------------------------
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
PRETRAINED_WEIGHTS = None
BATCH_SIZE = 64  # For general use; override with TRAINING_BATCH_SIZE if needed
