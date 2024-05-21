import numpy as np

import tensorflow as tf
from tensorflow.keras import models
import tensorflow_addons as tfa
from keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import shutil

from tqdm import tqdm

epoch_size = 100
batch_s = 32
input_shape = (224, 224, 3)
img_size = (224, 224)
dataset_directory = "train_split_500"
filtered_dataset_directory = "train_split_500"


def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


import os


def create_model(input_shape):
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling=None
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    return models.Model(inputs=base_model.input, outputs=out)


model = create_model(input_shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss()
)

for labels in tqdm(os.walk(dataset_directory), total=167):
    if labels[0] == dataset_directory:
        continue
    preds = []
    for i, img in enumerate(labels[2]):
        img_path = os.path.join(labels[0], img)
        img_array = preprocess_image(img_path)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array, verbose=0)
        preds.append(np.squeeze(pred))

    vector_size = len(preds)
    matrix = np.zeros((vector_size, vector_size))

    for i, itemA in enumerate(preds):
        for j, itemB in enumerate(preds):
            matrix[i, j] = np.dot(itemA, itemB) / (
                np.linalg.norm(itemA) * np.linalg.norm(itemB)
            )

    sims = np.zeros(vector_size, dtype=object)
    for i, (inn, img_path) in enumerate(zip(matrix, labels[2])):
        sum = 0
        for j in inn:
            sum += j
        sims[i] = (i, img_path, sum / vector_size)

    sorted_sims = sorted(sims, key=lambda x: x[2], reverse=True)

    label = labels[0].split("/")[1]
    target_directory = "/mnt/n/Bakis/train_split_300_filtered/" + label
    source_directory = "/mnt/n/Bakis/train_split_500/" + label
    os.mkdir(target_directory)
    j = 0
    for i, path, sim in sorted_sims:
        img_path = os.path.join(target_directory, path)
        if j < 300:
            shutil.copy(source_directory + "/" + path, img_path)
        else:
            break
        j += 1
