import tensorflow as tf
import numpy as np
from model import get_model
from dataset import get_dataset

SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=1e-6
)

checkpoint_path = f"checkpoints/ckpt_best"
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, save_best_only=True
)


train_dataset = get_dataset(True, SIZE, EPOCHS, BATCH_SIZE)
test_dataset = get_dataset(False, SIZE, 1, BATCH_SIZE)

base_model, model = get_model(SIZE)


def compile_and_train(m, epochs, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    m.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=len(train_dataset) // EPOCHS,
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
    model = compile_and_train(model, EPOCHS, ri)

model.save_weights(f"model{SIZE}.ckpt")
