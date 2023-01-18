# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path
import cv2
# -

from functions_custom import load_data, train_model, plot_output

# Simple image classification example: https://www.tensorflow.org/tutorials/images/classification

# # Set up

train_dir = Path("..", "data", "3_Train_Test_Folder", "train")
test_dir = Path("..", "data", "3_Train_Test_Folder", "test")

# +
BATCH_SIZE = 32
EPOCHS = 15

img_height = 180
img_width = 180
# -

train, test, num_classes = load_data(train_dir, test_dir, BATCH_SIZE, img_height, img_width)

# +
AUTOTUNE = tf.data.AUTOTUNE

train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test = test.cache().prefetch(buffer_size=AUTOTUNE)
# -

# # Model_01

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)


folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_01")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -

# # Model_02 - 2 conv layers

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_02")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -

# # Model_03 - 4 conv layers

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_03")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -

# # Model 04 - 2 fully connected layers (dropout)

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_04_dropout")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -
# # Model 4 (2 fully connected layers) original

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_04")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -

# # ~Model_05 5 conv layers


# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_05")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -

# # ~Model_06 6 conv layers

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_06")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# -
# # ~Model_07 13 conv layers

# +
model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

folder_path = Path("..", "outputs", f'batchsize_{BATCH_SIZE}', "model_07")

model = train_model(model, train, test, folder_path, EPOCHS)

plot_output(model, EPOCHS, BATCH_SIZE, folder_path)
# +
#layers.Dropout(.2)
# -


