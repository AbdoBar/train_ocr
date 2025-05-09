import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Levenshtein import distance as lev_distance

# ====================
# 1. Data Preparation
# ====================


def generate_synthetic_data(word_list, num_samples=100):
    """Generate synthetic training images using Arabic fonts"""
    images = []
    labels = []
    # Use actual Arabic font paths - EXAMPLE FOR LINUX:
    fonts = [
        '/usr/share/fonts/truetype/arabtype/Arabic_Transparent.ttf',  # Actual Arabic font
        '/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf'   # Common Arabic font
    ]
    
    for word in word_list:
        for _ in range(num_samples):
            try:
                # Load font with error handling
                font_size = np.random.randint(24, 48)
                font = ImageFont.truetype(np.random.choice(fonts), font_size)
                
                # Rest of the image generation code...
                
            except OSError as e:
                print(f"Font loading error: {e}")
                continue

    return np.array(images), np.array(labels)   

# Load vocabulary from Excel
#df = pd.read_excel('words class.xlsx', sheet_name='Sheet1')
#word_list = df['B'].tolist()
df = pd.read_excel('words class.xlsx', sheet_name='Sheet1', header=None)  # No headers
word_list = df.iloc[:, 1].tolist()  # Get second column (Arabic words)


# Generate synthetic dataset
X, y = generate_synthetic_data(word_list, num_samples=100)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2)

# ====================
# 2. CNN Model
# ====================

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((*input_shape, 1), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_model(X_train[0].shape, len(word_list))
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# ====================
# 3. NLP Correction
# ====================

class NLPCorrector:
    def __init__(self, vocabulary):
        self.vocab = vocabulary
        
    def correct(self, word):
        # Find closest match in vocabulary
        distances = [(v, lev_distance(word, v)) for v in self.vocab]
        return min(distances, key=lambda x: x[1])[0]

corrector = NLPCorrector(word_list)

# ====================
# 4. Inference Pipeline
# ====================

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def recognize_word(img_path):
    # Preprocess image
    processed_img = preprocess_image(img_path)
    
    # CNN Prediction
    pred_probs = model.predict(processed_img)
    pred_idx = np.argmax(pred_probs)
    raw_pred = label_encoder.inverse_transform([pred_idx])[0]
    
    # NLP Correction
    corrected = corrector.correct(raw_pred)
    
    return {
        'raw_prediction': raw_pred,
        'corrected_prediction': corrected,
        'confidence': float(np.max(pred_probs))
    }

# ====================
# 5. Testing Environment
# ====================
# Usage example:
result = recognize_word('test_image.tif')
print(f"Input Image: test_image.tif")
print(f"Raw Prediction: {result['raw_prediction']}")
print(f"Corrected Prediction: {result['corrected_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
