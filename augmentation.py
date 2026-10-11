
import tensorflow as tf


def build_augmentation():
    """Build mild augmentation for SEM defect images."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(
            0.02, fill_mode="reflect"
        ),
        tf.keras.layers.RandomZoom(
            0.05, fill_mode="reflect"
        ),
        tf.keras.layers.RandomContrast(0.05),
    ], name="sem_augmentation")


def augment_train_dataset(dataset):
    """Apply augmentation only to the training dataset."""
    augmentation = build_augmentation()

    def augment(images, labels):
        images = augmentation(images, training=True)
        images = tf.clip_by_value(images, 0.0, 255.0)
        return images, labels

    return dataset.map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )
