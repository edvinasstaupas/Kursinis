import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    # base_model.trainable = False
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation=None),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalization
    ])

    return model

input_shape = (224, 224, 3)

siamese_model = create_model(input_shape)

anchor_input = tf.keras.Input(input_shape)
positive_input = tf.keras.Input(input_shape)
negative_input = tf.keras.Input(input_shape)

anchor_embedding = siamese_model(anchor_input)
positive_embedding = siamese_model(positive_input)
negative_embedding = siamese_model(negative_input)

merged_vector = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
training_model = models.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)

# Create an embedding model for generating embeddings for single images
base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
base_model.trainable = False

m = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation=None),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalization
    ])
embedding_layer = siamese_model(anchor_input)
embedding_model = models.Model(inputs=anchor_input, outputs=embedding_layer)

# Save the embedding model
training_model.save('model/base_trainable/training_model.keras')
embedding_model.save('model/base_trainable/embedding_model.keras')