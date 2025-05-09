import os
import cv2
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM,
                                     Bidirectional, Lambda, BatchNormalization)
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Load data from Excel file
excel_path = 'words class.xlsx'  # Corrected file name
image_folder = 'images/'

# Read data (no headers, columns 0 and 1)
data = pd.read_excel(excel_path, header=None)
image_names = data[0].astype(str).values  # Ensure names are strings
labels = data[1].values

# Tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(labels)
sequences = tokenizer.texts_to_sequences(labels)
num_classes = len(tokenizer.word_index) + 1  # +1 for padding/blank

# Load images and verify existence
IMG_HEIGHT, IMG_WIDTH = 64, 256
X_img, valid_labels, label_lengths = [], [], []

for i, name in enumerate(image_names):
    image_path = os.path.join(image_folder, f"{name}.tif")  # Add .tif extension
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X_img.append(img)
        valid_labels.append(sequences[i])
        label_lengths.append(len(sequences[i]))
    else:
        print(f"⚠️ Image not found: {image_path}")

if not X_img:
    raise ValueError("No valid images found!")

X_img = np.array(X_img).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
max_label_len = max(label_lengths)
Y_encoded = pad_sequences(valid_labels, maxlen=max_label_len, padding='post')

# Split data
X_train, X_val, Y_train, Y_val, label_len_train, label_len_val = train_test_split(
    X_img, Y_encoded, label_lengths, test_size=0.2, random_state=42
)

# Correct input length calculation
time_steps = IMG_WIDTH // 4  # 256//4=64
train_input_len = np.ones((len(X_train), 1)) * time_steps
val_input_len = np.ones((len(X_val), 1)) * time_steps
label_len_train = np.expand_dims(label_len_train, 1)
label_len_val = np.expand_dims(label_len_val, 1)

# CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

# Build model
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input_img')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64)
x = Reshape(target_shape=new_shape)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dense(num_classes, activation='softmax')(x)

labels = Input(name='labels', shape=(max_label_len,), dtype='int32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss={'ctc': lambda y_true, y_pred: y_pred})

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    x=[X_train, Y_train, train_input_len, label_len_train],
    y=np.zeros(len(X_train)),
    validation_data=([X_val, Y_val, val_input_len, label_len_val], np.zeros(len(X_val))),
    epochs=20,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)

# Prediction model
prediction_model = Model(inputs=input_img, outputs=x)

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * (pred.shape[1])
    pred = np.argmax(pred, axis=-1)
    texts = []
    for p in pred:
        p = [k for k, g in groupby(p)]
        p = [x for x in p if x != 0]  # Remove blanks/padding
        text = tokenizer.sequences_to_texts([p])[0]
        texts.append(text)
    return texts

# Test function
def test_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
    pred = prediction_model.predict(img)
    decoded = decode_prediction(pred)
    return decoded[0]

# Example test
print(test_image('images/2.tif'))  # Test with provided sample image
