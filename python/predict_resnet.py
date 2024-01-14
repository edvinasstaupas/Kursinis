import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import datetime
from tqdm import tqdm

test_dataset_dir = 'test_split_25'
filename = 'matrix/resnet/matrix.txt'
filename2 = 'matrix/resnet/matrix.npy'

batch_s = 1

def load_custom_dataset(directory, image_size=(224, 224)):
    test_dataset = image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=image_size,
        shuffle=False,
    )

    return test_dataset

print("Started loading dataset: ", datetime.datetime.now())
test_dataset = load_custom_dataset(test_dataset_dir, image_size=(224, 224))
print("Loaded dataset: ", datetime.datetime.now())

input_shape = (224, 224, 3)
# Load the model
print("Started loading model: ", datetime.datetime.now())
base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
print("Loaded model: ", datetime.datetime.now())
print("Started predicting: ", datetime.datetime.now())
test_dataset_batched = test_dataset.batch(batch_s)
pred = base_model.predict(test_dataset_batched, batch_size=batch_s)
print("Predicted: ", datetime.datetime.now())

vector_size = len(pred)
matrix = np.zeros((vector_size, vector_size))

for i, itemA in enumerate(pred):
    itemA = itemA.reshape((49, 2048))
    for j, itemB in enumerate(pred):
        itemB = itemB.reshape((49, 2048))
        matrix[i, j] = np.dot(itemA.flatten(), itemB.flatten()) / (
            np.linalg.norm(itemA.flatten()) * np.linalg.norm(itemB.flatten())
)

print("Vector:")
print(pred)

print("\nMatrix:")
print(matrix)

# Save the matrix to a text file
np.savetxt(filename, matrix)
np.save(filename2, matrix)