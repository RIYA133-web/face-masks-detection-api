import tensorflow as tf

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=32
)

print("Dataset loaded successfully")