import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import datetime
from tqdm import tqdm
import tensorflow_addons as tfa
import os

test_dataset_dir = 'test_split_25'
embedding_model_dir = 'model/base_trainable/embedding_model.keras'
checkpoint_path = "training_1/checkpoints_new/cp-0028.ckpt"
filename = 'matrix/final/matrix.txt'
filename2 = 'matrix/final/matrix.npy'

checkpoint_dir = os.path.dirname(checkpoint_path)
batch_s = 32

def load_custom_dataset(directory, image_size=(224, 224)):
    test_dataset = image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=image_size,
        shuffle=False
    )

    return test_dataset

print("Started loading dataset: ", datetime.datetime.now())
test_dataset = load_custom_dataset(test_dataset_dir, image_size=(224, 224))
print("Loaded dataset: ", datetime.datetime.now())

# Load the model
print("Started loading model: ", datetime.datetime.now())
embedding_model = tf.keras.models.load_model(embedding_model_dir, safe_mode=False)
embedding_model.compile(optimizer=Adam(learning_rate=0.001), loss=tfa.losses.TripletSemiHardLoss(soft=False))
embedding_model.load_weights(checkpoint_path)
print("Loaded model: ", datetime.datetime.now())
print("Started predicting: ", datetime.datetime.now())
test_dataset_batched = test_dataset.batch(batch_s)
pred = embedding_model.predict(test_dataset_batched, batch_size=batch_s)
print("Predicted: ", datetime.datetime.now())

vector_size = len(pred)
matrix = np.zeros((vector_size, vector_size))
matrix_2 = np.zeros((vector_size, vector_size))

for i, itemA in tqdm(enumerate(pred), total = vector_size):
    for j, itemB in enumerate(pred):
        matrix[i, j] = np.dot(itemA, itemB) / (np.linalg.norm(itemA) * np.linalg.norm(itemB))

print("Vector:")
print(pred)

print("\nMatrix:")
print(matrix)

print(matrix)

filename3 = 'matrix/final/pred.txt'
# Save the matrix to a text file
np.savetxt(filename3, pred)
np.savetxt(filename, matrix)
np.save(filename2, matrix)
