import tensorflow as tf
import numpy as np
from utils import get_decomp

# from sparse_conv2d import SparseConv2D, clone_function
from custom_resnet import ResNet50


# copied from Kaggle notebook, will cite in write-up
def get_resnet(ws_path=None):
    base_model = tf.keras.applications.resnet.ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    # base_model = tf.keras.models.clone_model(base_model, clone_function=clone_function)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.UpSampling2D(size=(7, 7), interpolation="bilinear"))
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(100, activation="softmax"))
    ins = np.zeros(shape=(1, 32, 32, 3))
    _ = model(ins)
    base_model.trainable = False
    if ws_path:
        model.load_weights(ws_path)
    return base_model, model


def _decomp_resnet(ranks):
    ins = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.UpSampling2D(size=(7, 7), interpolation="bilinear")(ins)
    x = ResNet50(ranks=ranks, include_top=False, weights=None, input_tensor=x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(100, activation="softmax")(x)
    model = tf.keras.Model(inputs=ins, outputs=x)
    return model


def get_decomp_resnet(ranks, spars, ws_path):
    model = _decomp_resnet(ranks)
    name_set = set([l.name for l in model.layers])
    base_model, model2 = get_resnet(ws_path)
    bias3 = [
        l
        for l in base_model.layers
        if isinstance(l, tf.keras.layers.Conv2D) and l.kernel.shape[0] == 3
    ]
    for i, l in enumerate(bias3):
        rank = ranks[i]
        if rank == 0:
            model.get_layer(l.name).set_weights(l.get_weights())
            name_set.remove(l.name)
            continue
        spar = spars[i]
        core, first, last, sp = get_decomp(
            l.kernel.numpy(), (2, 3), [rank, rank], 100 - spar
        )
        model.get_layer(l.name + "_first").set_weights([np.expand_dims(first, [0, 1])])
        model.get_layer(l.name + "_core").set_weights([core])
        model.get_layer(l.name + "_last").set_weights([np.expand_dims(last.T, [0, 1])])
        model.get_layer(l.name + "_sp").set_weights([sp, l.bias.numpy()])
        name_set.remove(l.name + "_sp")
        name_set.remove(l.name + "_first")
        name_set.remove(l.name + "_core")
        name_set.remove(l.name + "_last")

    for l in base_model.layers:
        if l not in bias3 and "input" not in l.name:
            model.get_layer(l.name).set_weights(l.get_weights())
            name_set.remove(l.name)
    model2_names = ["dense_2", "batch_normalization_1", "dense_3"]
    model_names = ["dense", "batch_normalization", "dense_1"]
    for m2, m in zip(model2_names, model_names):
        model.get_layer(m).set_weights(model2.get_layer(m2).get_weights())
    return model
