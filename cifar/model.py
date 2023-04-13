import tensorflow as tf
import numpy as np

# from sparse_conv2d import SparseConv2D, clone_function
from custom_resnet import ResNet50


# weights from model
def resnet_wfm():
    model = tf.keras.models.load_model("model/resnet.h5")
    model.save_weights("model/resnet_weights.h5")


# copied from Kaggle notebook, will cite in write-up
def get_resnet():
    base_model = tf.keras.applications.resnet.ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
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
    model.load_weights("model/resnet_weights.h5")
    return base_model, model


def get_decomp_resnet(ranks):
    ins = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.UpSampling2D(size=(7, 7), interpolation="bilinear")(ins)
    x = ResNet50(ranks=ranks, include_top=False, weights=None, input_tensor=x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(100, activation="softmax")(x)
    model = tf.keras.Model(inputs=ins, outputs=x)
    model.load_weights("model/resnet_weights.h5", by_name=True)
    return model
