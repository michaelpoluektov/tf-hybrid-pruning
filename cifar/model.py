import tensorflow as tf
import numpy as np
from sparse_conv2d import SparseConv2D, clone_function


# copied from Kaggle notebook, will cite in write-up
def get_resnet():
    model = tf.keras.models.load_model("model/my_model.h5")
    return model.layers[1], model
