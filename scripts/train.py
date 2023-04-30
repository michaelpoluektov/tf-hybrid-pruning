import tensorflow as tf
import sys

sys.path.append("../src")
from model import get_resnet
from dataset import get_dataset
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


def train(base_model, model, train_ds, test_ds, reg, filename, freeze=False):
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    if reg:
        for l in base_model.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                base_model.add_loss(lambda layer=l: reg(l.kernel))
        for l in model.layers:
            if isinstance(l, tf.keras.layers.Dense):
                model.add_loss(lambda layer=l: reg(l.kernel))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if freeze:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "categorical_crossentropy"],
    )
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=100,
        verbose=1,
        callbacks=[learning_rate_reduction, early_stopping],
    )
    model.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "categorical_crossentropy"],
    )
    model.save_weights(f"../models/{filename}.h5")


if __name__ == "__main__":
    SIZE = 224
    # reg = tf.keras.regularizers.L2(3e-4)
    # reg = tf.keras.regularizers.L1(2e-5)
    reg = None
    base_model, model = get_resnet()
    test_ds, val_ds = get_dataset(False, SIZE, 1, 16, True)
    train_ds = get_dataset(True, SIZE, 1, 16, False)
    # train_bn(base_model, model, train_ds, test_ds)
    train(base_model, model, train_ds, test_ds, reg, "resnet_bn_finetune", True)
