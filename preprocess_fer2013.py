import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fer2013.csv")

# Extract pixels and labels
pixels = df["pixels"].values
labels = df["emotion"].values

# Convert pixel strings to arrays
X = np.array([
    np.array(p.split(), dtype="float32").reshape(48, 48)
    for p in pixels
])

y = labels

print("Raw X shape:", X.shape)
print("Raw y shape:", y.shape)

# Normalize pixel values (0–255 → 0–1)
X = X / 255.0

# Add channel dimension for CNN (grayscale)
X = X.reshape(-1, 48, 48, 1)

print("Processed X shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
