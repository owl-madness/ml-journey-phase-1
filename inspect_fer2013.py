import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("fer2013.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nEmotion label counts:")
print(df["emotion"].value_counts())

# Inspect one image
pixels = df.iloc[0]["pixels"]
emotion = df.iloc[0]["emotion"]

pixel_array = np.array(pixels.split(), dtype="float32")
image = pixel_array.reshape(48, 48)

print("\nSingle image shape:", image.shape)
print("Emotion label:", emotion)
