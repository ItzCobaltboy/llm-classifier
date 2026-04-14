print('='*50)
print("\t\tImporting Libraries")
print('='*50, '\n')
import sys
import subprocess
import importlib

def install_if_missing(packages):
    """
    Checks if packages are installed. If not, installs them via pip.
    'packages' should be a dictionary of {import_name: pip_name}.
    """
    for import_name, pip_name in packages.items():
        try:
            # Try to import the library
            importlib.import_module(import_name)
            print(f"✅ {import_name} is already installed.")
        except ImportError:
            # If it fails, install it using the current Python environment's pip
            print(f"⏳ {import_name} not found. Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ Successfully installed {pip_name}.")

required_packages = {
    'torch': 'torch',
    'accelerate': 'accelerate',
    'shap': 'shap',
    'datasets': 'datasets',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'transformers': 'transformers',
    'sklearn': 'scikit-learn',
    'textstat': 'textstat',
    'tqdm': 'tqdm'
}

# Run the installation check
print("--- Checking Environment Setup ---")
install_if_missing(required_packages)
print("--- Setup Complete. Ready to Import! ---")

import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ── HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

# ── Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# ── Stylometric helpers
import textstat
import re
from collections import Counter
from tqdm.auto import tqdm

# ── Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

##################################################################################################################################################
