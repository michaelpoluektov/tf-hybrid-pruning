import tensorflow as tf
import numpy as np
from sparse_conv2d import SparseConv2D, clone_function


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
