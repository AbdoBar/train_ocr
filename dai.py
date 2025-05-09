import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load labels from Excel
df = pd.read_excel('words class.xlsx', sheet_name='Sheet1')
labels = df['B'].tolist()  # Arabic words
label_to_index = {word: idx for idx, word in enumerate(labels)}

# Load and preprocess images
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Grayscale
        img = cv2.resize(img, (128, 128))  # Resize to fixed dimensions
        img = img / 255.0  # Normalize
        images.append(img)
    return np.array(images)

# Example: Assuming image filenames correspond to IDs (1.tif, 2.tif, etc.)
image_paths = [f'data/{i}.tif' for i in df['A']]
X = load_images(image_paths)
y = np.array([label_to_index[label] for label in labels])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

from Levenshtein import distance as lev_dist

def correct_prediction(predicted_word):
    # Find the closest word in the label list
    distances = [(word, lev_dist(predicted_word, word)) for word in labels]
    return min(distances, key=lambda x: x[1])[0]

# Example usage
def predict_and_correct(image_path):
    img = load_images([image_path])[0][np.newaxis, ..., np.newaxis]
    pred_idx = np.argmax(model.predict(img))
    predicted_word = labels[pred_idx]
    corrected_word = correct_prediction(predicted_word)
    return corrected_word

# Test on a sample image (e.g., 2.tif)
print(predict_and_correct('data/2.tif'))
