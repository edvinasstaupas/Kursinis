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


def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


def _normalize_img(img, label):
    img = preprocess_input(img)
    return (img, label)


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

from tqdm import tqdm

test_dataset_dir = "test_split_clean_50"
filename2 = "matrix/resnet-similarity-gldv-quad_clean-dataset/matrix.npy"

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
    "training_1/checkpoints/resnet-similarity-gldv_quad_clean/cp-0030.ckpt"
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

np.save(filename2, matrix)
