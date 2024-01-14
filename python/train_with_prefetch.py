import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tqdm import tqdm
import tensorflow_addons as tfa
import os
import logging

# Configure the logging module
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('logfile.txt')  # Log to file
    ]
)

epoch_size = 100
batch_s = 64
seed = 1337
rng = np.random.default_rng(seed)
dataset_directory = '/mnt/a/Kursinis/train_split_25'
cache_directory = '/mnt/a/Kursinis/image_cache'
model_dir = 'model/base_trainable/training_model.keras'

def create_image_label_map(dir):
    train_map = {}
    val_map = {}
    for root, dirs, files in os.walk(dir):
        for directory in dirs:
            directory_path = os.path.join(root, directory)
            files_in_directory = os.listdir(directory_path)

            # Split files into training and validation sets
            split_index = 20

            img_list_train = files_in_directory[:split_index]
            img_list_val = files_in_directory[split_index:]
            train_map[directory] = img_list_train
            val_map[directory] = img_list_val

    return train_map, val_map

def preprocess_image(image_path):
    # Load and preprocess a single image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def create_cache(dir, ds, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache = {}
    for label, label_paths in tqdm(ds.items(), desc="Caching Images"):
        for img_path in label_paths:
            image_key = f"{label}_{img_path}"
            cache_path = os.path.join(cache_dir, f"{image_key}.npy")

            if not os.path.exists(cache_path):
                image = preprocess_image(os.path.join(dir, label, img_path))
                np.save(cache_path, image)
            
            cache[image_key] = cache_path
    return cache

def triplet_generator_with_cache(ds, cache):
    # Create positive and negative pairs for the Siamese network
    keys = [*ds]
    num_triplets = len(keys)

    i = 0  # Start index for yielding batches
    while True:
        i += 1
        if i >= num_triplets:
            i = 0
            np.random.shuffle(ds)

        for label, label_paths in ds.items():
            for anchor_path in label_paths:
                anchor_key = f"{label}_{anchor_path}"
                anchor_image = np.load(cache[anchor_key])

                anchor_label = float(label)  # Convert label to float

                # Select a positive image with the same label but different image
                positive_path = anchor_path
                while positive_path == anchor_path:
                    positive_path = np.random.choice(label_paths)
                positive_key = f"{label}_{positive_path}"
                positive_image = np.load(cache[positive_key])

                # Select a negative image with a different label and different image
                negative_path = anchor_path
                while negative_path == anchor_path:
                    negative_label = np.random.choice(keys)
                    negative_path = np.random.choice(ds[negative_label])
                negative_key = f"{negative_label}_{negative_path}"
                negative_image = np.load(cache[negative_key])

                yield (anchor_image, positive_image, negative_image), anchor_label

# Load the model
logging.info("Started loading model")
model = tf.keras.models.load_model(model_dir, safe_mode=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss=tfa.losses.TripletSemiHardLoss(soft=False))
logging.info("Loaded model")

# Load custom dataset
logging.info("Loading datasets")

train_map, val_map = create_image_label_map(dataset_directory)
logging.info("Created image label map")
logging.info("Started creating cache")
train_cache = create_cache(dataset_directory, train_map, cache_directory)
val_cache = create_cache(dataset_directory, val_map, cache_directory)
logging.info("Created cache")

# Create triplet datasets for training and validation
train_triplet_dataset = tf.data.Dataset.from_generator(
    lambda: triplet_generator_with_cache(train_map, train_cache),
    output_signature=(
        (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size = batch_s, num_parallel_calls = tf.data.AUTOTUNE, deterministic = False).prefetch(buffer_size=tf.data.AUTOTUNE)

val_triplet_dataset = tf.data.Dataset.from_generator(
    lambda: triplet_generator_with_cache(val_map, val_cache),
    output_signature=(
        (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(batch_size = batch_s, num_parallel_calls = tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

logging.info("Loaded datasets")

checkpoint_path = "training_1/checkpoints_new/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

logging.info("Started fitting")

# Calculate dataset lengths
def count_samples(img_map):
    total_samples = 0
    for label_paths in img_map.values():
        total_samples += len(label_paths)
    return total_samples

train_len = count_samples(train_map)
val_len = count_samples(val_map)

# Now you can use these generators in the fit method
model.fit(
    train_triplet_dataset,
    steps_per_epoch=train_len // batch_s + 1,
    validation_data=val_triplet_dataset,
    validation_steps=val_len // batch_s + 1,
    epochs=epoch_size,
    metrics=["accuracy", "loss"],
    callbacks=[cp_callback, early_stopping]
)
logging.info("Finished fitting")
