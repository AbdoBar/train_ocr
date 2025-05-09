import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM,
                                     Bidirectional, Lambda, BatchNormalization)
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import groupby

# === Configuration ===
IMG_HEIGHT, IMG_WIDTH = 64, 256
EXCEL_PATH = 'Groundtruth-Unicode.xlsx'
IMAGE_FOLDER = 'images/'

# === Load and preprocess data ===
data = pd.read_excel(EXCEL_PATH, header=None)
image_names = data[0].values
text_labels = data[1].values

# Tokenize characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text_labels)
sequences = tokenizer.texts_to_sequences(text_labels)
num_classes = len(tokenizer.word_index) + 1  # +1 for CTC blank label (index 0)

# Load images and labels
X_img, valid_labels, label_lengths = [], [], []

for i, name in enumerate(image_names):
    image_path = os.path.join(IMAGE_FOLDER, name)
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X_img.append(img)
        valid_labels.append(sequences[i])
        label_lengths.append(len(sequences[i]))
    else:
        print(f"⚠️ Image not found: {image_path}")

X_img = np.array(X_img).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
max_label_len = max(label_lengths)
Y_encoded = pad_sequences(valid_labels, maxlen=max_label_len, padding='post')

# === Split dataset ===
X_train, X_val, Y_train, Y_val, label_len_train, label_len_val = train_test_split(
    X_img, Y_encoded, label_lengths, test_size=0.2, random_state=42
)

train_input_len = np.ones((len(X_train), 1)) * (IMG_WIDTH // 4)  # Due to 2 pooling layers
val_input_len = np.ones((len(X_val), 1)) * (IMG_WIDTH // 4)
label_len_train = np.expand_dims(label_len_train, 1)
label_len_val = np.expand_dims(label_len_val, 1)

# === Define CTC loss ===
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

# === Build Model ===
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input_img')

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

# Reshape for RNN
new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)
x = Reshape(target_shape=new_shape)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
y_pred = Dense(num_classes, activation='softmax', name='y_pred')(x)

# Inputs for loss
label_input = Input(name='labels', shape=(max_label_len,), dtype='int32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')

# Loss output
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [y_pred, label_input, input_length, label_length]
)

# Compile model
# Compile model
model = Model(inputs=[input_img, label_input, input_length, label_length], outputs=loss_out)
model.compile(optimizer=Adam(), loss={'ctc': lambda y_true, y_pred: y_pred})  # Fix here
model.summary()

# === Train Model ===
model.fit(
    x=[X_train, Y_train, train_input_len, label_len_train],
    y=np.zeros(len(X_train)),
    validation_data=([X_val, Y_val, val_input_len, label_len_val], np.zeros(len(X_val))),
    epochs=20,
    batch_size=16
)

# === Prediction Model ===
prediction_model = Model(inputs=input_img, outputs=y_pred)

# === Decode function ===
def decode_prediction(pred):
    pred_indices = np.argmax(pred, axis=-1)
    texts = []
    for p in pred_indices:
        # Remove repeats and blank token (0)
        p = [k for k, _ in groupby(p) if k != 0]
        text = tokenizer.sequences_to_texts([p])[0]
        texts.append(text)
    return texts

# === Predict on single image ===
def test_image(img_path):
    if not os.path.exists(img_path):
        return "Image not found"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
    pred = prediction_model.predict(img, verbose=0)
    decoded = decode_prediction(pred)
    return decoded[0] if decoded else ""

# === Example Usage ===
print(test_image('images/AHTD3A0001_Para2_1.tif'))

