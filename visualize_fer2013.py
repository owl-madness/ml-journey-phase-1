import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Emotion labels
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Load dataset
df = pd.read_csv("fer2013.csv")

# Show first 9 images
plt.figure(figsize=(8, 8))

for i in range(9):
    pixels = df.iloc[i]["pixels"]
    emotion = df.iloc[i]["emotion"]
    
    image = np.array(pixels.split(), dtype="float32").reshape(48, 48)
    
    plt.subplot(3, 3, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(emotion_map[emotion])
    plt.axis("off")

plt.tight_layout()
plt.show()
