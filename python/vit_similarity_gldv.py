import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from vit_keras import vit
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tqdm import tqdm
import io


def create_model():
    image_size = 224
    model = vit.vit_b32(
        image_size=image_size,
        activation=None,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    x = model.output
    x = layers.Dense(256)(x)
    out = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs=model.input, outputs=out)


model = create_model()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss()
)

input_shape = (224, 224, 3)
img_size = (224, 224)
batch_size = 64
dataset_directory = "train_split_500"


def _normalize_img(img, label):
    img = tf.keras.layers.Rescaling(1.0 / 255)(img)
    return (img, label)


dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=None,
    image_size=img_size,
    shuffle=False,
)

train_length = int(len(dataset) * 0.8)
dataset = dataset.shuffle(1024)
train_dataset = dataset.take(train_length)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.map(_normalize_img)

validation_dataset = dataset.skip(train_length)
validation_dataset = validation_dataset.batch(batch_size)
validation_dataset = validation_dataset.map(_normalize_img)

print(len(train_dataset))
print(len(validation_dataset))

checkpoint_path = "training_1/checkpoints/vit-gldv-similarity/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=[cp_callback],
)

test_dataset_dir = "test_split_500"
filename = "matrix/vit-gldv-similarity/matrix.txt"
filename2 = "matrix/vit-gldv-similarity/matrix.npy"

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

model.load_weights("training_1/checkpoints/vit-gldv-similarity/cp-0010.ckpt")
test_dataset = (
    test_dataset.map(_normalize_img).batch(batch_size).prefetch(tf.data.AUTOTUNE)
)

pred = model.predict(test_dataset, batch_size=batch_size)


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

filename3 = "matrix/vit-gldv-similarity/pred.txt"

np.savetxt(filename3, pred)
np.savetxt(filename, matrix)
np.save(filename2, matrix)
