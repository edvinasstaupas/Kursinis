import numpy as np

import tensorflow as tf
from tensorflow.keras import models
import tensorflow_addons as tfa
from keras.applications.resnet_v2 import preprocess_input
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import io

epoch_size = 100
batch_s = 32
seed = 1337
input_shape = (224, 224, 3)
img_size = (224, 224)
rng = np.random.default_rng(seed)
dataset_directory = "train_split_clean"
checkpoint_path = (
    "training_1/checkpoints/resnet-similarity-gldv_quad_clean/cp-{epoch:04d}.ckpt"
)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


def _normalize_img(img, label):
    img = preprocess_input(img)
    return (img, label)


import os
import math
import random

cache_directory = "/mnt/e/Kursinis/image_cache"


def create_image_label_map(dir):
    train_map = {}
    val_map = {}
    for root, dirs, files in os.walk(dir):
        for directory in dirs:
            directory_path = os.path.join(root, directory)
            files_in_directory = os.listdir(directory_path)

            split_index = math.floor(len(files_in_directory) * 0.8)

            img_list_train = files_in_directory[:split_index]
            img_list_val = files_in_directory[split_index:]
            train_map[directory] = img_list_train
            val_map[directory] = img_list_val

    return train_map, val_map


def create_cache(dir, ds, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache = {}
    for label, label_paths in tqdm(ds.items(), desc="Caching Images"):
        for img_path in label_paths:
            image_key = img_path
            cache_path = os.path.join(cache_dir, f"{image_key}.npy")

            if not os.path.exists(cache_path):
                image = preprocess_image(os.path.join(dir, label, img_path))
                np.save(cache_path, image)

            cache[image_key] = cache_path
    return cache


def pair_values(map_of_lists):
    paired_values = []
    for key, values in map_of_lists.items():
        random.shuffle(values)
        for i in range(0, len(values), 4):
            if i + 3 >= len(values):
                break
            paired_values.append(
                (values[i], values[i + 1], values[i + 2], values[i + 3], key)
            )
    return paired_values


def generator_with_cache(ds, cache):
    while True:
        paired_ds = pair_values(ds)
        random.shuffle(paired_ds)

        for item1, item2, item3, item4, label in paired_ds:
            yield (np.load(cache[item1]), label)
            yield (np.load(cache[item2]), label)
            yield (np.load(cache[item3]), label)
            yield (np.load(cache[item4]), label)


train_map, val_map = create_image_label_map(dataset_directory)

train_cache = create_cache(dataset_directory, train_map, cache_directory)
val_cache = create_cache(dataset_directory, val_map, cache_directory)

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator_with_cache(train_map, train_cache),
    output_signature=(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
).batch(batch_s)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: generator_with_cache(val_map, val_cache),
    output_signature=(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
).batch(batch_s)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)


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


def count_samples(img_map):
    total_samples = 0
    for label_paths in img_map.values():
        total_samples += len(label_paths)
    return total_samples


train_len = count_samples(train_map)
val_len = count_samples(val_map)


history = model.fit(
    train_dataset,
    steps_per_epoch=train_len // batch_s + 1,
    validation_data=validation_dataset,
    validation_steps=val_len // batch_s + 1,
    callbacks=[cp_callback],
    epochs=epoch_size,
)

from tqdm import tqdm

test_dataset_dir = "test_split_clean_50"
filename2 = "matrix/resnet-similarity-gldv-quad_v2/matrix.npy"

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dataset_dir,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=None,
    image_size=img_size,
    shuffle=False,
)

test_dataset = (
    test_dataset.map(_normalize_img).batch(batch_s).prefetch(tf.data.AUTOTUNE)
)

model.load_weights(
    "training_1/checkpoints/resnet-similarity-gldv_quad_clean/cp-0045.ckpt"
)
pred = model.predict(test_dataset, batch_size=batch_s)

from datetime import datetime

np.savetxt(
    "vecs" + datetime.now().strftime("%m_%d_%YT%H:%M:%S") + ".tsv", pred, delimiter="\t"
)

out_m = io.open(
    "meta" + datetime.now().strftime("%m_%d_%YT%H:%M:%S") + ".tsv",
    "w",
    encoding="utf-8",
)
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()

vector_size = len(pred)
matrix = np.zeros((vector_size, vector_size))
matrix_2 = np.zeros((vector_size, vector_size))

for i, itemA in tqdm(enumerate(pred), total=vector_size):
    for j, itemB in enumerate(pred):
        matrix[i, j] = np.dot(itemA, itemB) / (
            np.linalg.norm(itemA) * np.linalg.norm(itemB)
        )

print("Vector:")
print(pred)

filename3 = "matrix/resnet-similarity-gldv-quad_clean_45_second-test/pred.txt"

np.save(filename2, matrix)
