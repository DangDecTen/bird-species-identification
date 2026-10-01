"""Data loading: folder-per-class splits -> tf.data pipelines."""
from pathlib import Path

import keras
import tensorflow as tf

IMG_SIZE = 224


def load_split(split_dir, split, batch_size=32, img_size=IMG_SIZE, shuffle=None, seed=1337):
    """Load data/splits/<split>/<class>/*.jpg.

    Returns (dataset, class_names). Images are float32 RGB in [0, 255], resized
    with bilinear interpolation to img_size x img_size WITHOUT keeping aspect
    ratio (stretched). The Flutter app must do the same.
    Class index = alphabetical order of folder names.
    """
    if shuffle is None:
        shuffle = split == "train"
    ds = keras.utils.image_dataset_from_directory(
        Path(split_dir) / split,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(img_size, img_size),
        interpolation="bilinear",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return ds, ds.class_names


def build_augmentation():
    """Training-time augmentation (kept OUT of the exported model)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom((-0.25, 0.1)),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
            keras.layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


def prepare(ds, augment=False):
    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
