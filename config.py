# Hyperparameter search space for Optuna
SEARCH_SPACE = {
    'learning_rate': {
        'type': 'float',
        'low': 1e-5,
        'high': 1e-1,
        'log': True
    },
    'batch_size': {
        'type': 'categorical',
        'choices': [32, 64, 128, 256]
    },
    'optimizer': {
        'type': 'categorical',
        'choices': ['Adam', 'SGD', 'AdamW']
    },
    'weight_decay': {
        'type': 'float',
        'low': 1e-6,
        'high': 1e-2,
        'log': True
    },
    'dropout': {
        'type': 'float',
        'low': 0.0,
        'high': 0.6,
        'log': False
    }
}

# Training settings
NUM_EPOCHS = 10
MAX_TRIALS = 10
TIMEOUT = None  # No timeout - complete all trials
NUM_CLASSES = 10

# Device settings
USE_MPS = True  # Use Apple Silicon GPU if available

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]