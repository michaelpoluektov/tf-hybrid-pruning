import tensorflow as tf
import numpy as np
from model import get_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ]
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=1e-6
)

checkpoint_path = f"checkpoints/ckpt_best"
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, save_best_only=True
)


base_model, model = get_model(SIZE)

def get_dataset(x, y, r=True):
    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .batch(BATCH_SIZE)
        .map(lambda x, y: (tf.image.resize(x, [SIZE, SIZE]), y))
        .map(lambda x, y: (data_augmentation(x) if r else x, y))
        .prefetch(tf.data.AUTOTUNE)
        .repeat(1 if not r else EPOCHS)
    )


train_dataset = get_dataset(x_train, y_train)
test_dataset = get_dataset(x_test, y_test, False)


def compile_and_train(m, epochs, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    m.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=test_dataset,
        callbacks=[learning_rate_reduction, ckpt_cb],
    )
    return model


l = [0, 100, 200, 300, len(base_model.layers)]
r = [0.005, 0.001, 0.0005, 0.0003, 0.0001]

for li, ri in zip(l, r):
    for layer in base_model.layers[-li:]:
        if layer.__class__.__name__.lower() != "batchnormalization":
            layer.trainable = True
    model = compile_and_train(model, 5, ri)

model.save_weights(f"model{SIZE}.ckpt")
