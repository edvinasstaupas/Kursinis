import numpy as np

import tensorflow as tf
from keras.applications.resnet_v2 import preprocess_input
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import io

epoch_size = 100
batch_s = 32
seed = 1337
rng = np.random.default_rng(seed)
checkpoint_path = "training_1/checkpoints/resnet-numbers-similarity/cp-{epoch:04d}.ckpt"

input_shape = (32, 32, 3)
img_size = (32, 32)


def _normalize_img(img, label):
    img = tf.image.resize(img, img_size)
    img = tf.image.grayscale_to_rgb(img)
    img = preprocess_input(img)
    return (img, label)


dataset, test_dataset = tfds.load(
    name="mnist", split=["train", "test"], as_supervised=True
)

train_length = int(len(dataset) * 0.8)
dataset = dataset.shuffle(1024)
train_dataset = dataset.take(train_length)
train_dataset = train_dataset.batch(batch_s)
train_dataset = train_dataset.map(_normalize_img)

validation_dataset = dataset.skip(train_length)
validation_dataset = validation_dataset.batch(batch_s)
validation_dataset = validation_dataset.map(_normalize_img)

print(len(train_dataset))
print(len(validation_dataset))

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", patience=3, restore_best_weights=True
)


def create_model(input_shape):
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False, pooling="avg", weights=None, input_shape=input_shape
    )
    x = base_model.output
    x = tf.keras.layers.Dense(256, activation=None)(x)
    out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return tf.keras.Model(inputs=base_model.input, outputs=out)


model = create_model(input_shape)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss()
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    callbacks=[cp_callback, early_stopping],
    epochs=epoch_size,
)

from tqdm import tqdm

max_per_label = 892
test_dataset = (
    test_dataset.map(_normalize_img).batch(batch_s).prefetch(tf.data.AUTOTUNE)
)


label_counts = np.zeros(10)
label_images = [[] for _ in range(10)]
for images, labels in test_dataset:
    for i in range(len(labels)):
        label_index = labels[i].numpy()
        label_counts[label_index] += 1
        label_images[label_index].append((images[i], labels[i]))

sorted_filtered_dataset = []
target_count = 892


def test_generator():
    for i in range(10):
        label_image = label_images[i][:target_count]
        for image, label in label_image:
            yield (image, label)


test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    ),
)
test_dataset = test_dataset.batch(32)


def count_class(counts, batch, num_classes=10):
    _, labels = batch
    for i in range(num_classes):
        cc = tf.cast(labels == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)
    return counts


initial_state = dict((i, 0) for i in range(10))
counts = test_dataset.reduce(initial_state=initial_state, reduce_func=count_class)

model.load_weights(
    "training_1/checkpoints/resnet-numbers-similarity/cp-0009.ckpt"
).expect_partial()
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

filename = "matrix/numbers-similarity/matrix.txt"
filename2 = "matrix/numbers-similarity/matrix.npy"

np.savetxt(filename, matrix)
np.save(filename2, matrix)
