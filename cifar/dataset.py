import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)


def _get_dataset(is_train, size, epochs, batch_size, xy):
    return (
        tf.data.Dataset.from_tensor_slices(xy)
        .batch(batch_size)
        .map(lambda x, y: (tf.image.resize(x, [size, size]), y))
        .map(lambda x, y: (data_augmentation(x) if is_train else x, y))
        .prefetch(tf.data.AUTOTUNE)
        .repeat(epochs)
    )


def get_dataset(is_train, size, epochs, batch_size, split=False):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )

    xy = (x_train if is_train else x_test, y_train if is_train else y_test)
    if split:
        half = len(xy[0]) // 2
        test_xy = (xy[0][:half], xy[1][:half])
        val_xy = (xy[0][half:], xy[1][half:])
        test_ds = _get_dataset(is_train, size, epochs, batch_size, test_xy)
        val_ds = _get_dataset(is_train, size, epochs, batch_size, val_xy)
        return (test_ds, val_ds)
    return _get_dataset(is_train, size, epochs, batch_size, xy)
