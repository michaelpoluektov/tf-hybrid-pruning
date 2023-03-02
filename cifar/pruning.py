from utils import prune_all
import tensorflow as tf
from model import get_model
from dataset import get_dataset
import tensorflow_probability as tfp
from tensorflow.keras import mixed_precision
import pickle

mixed_precision.set_global_policy("mixed_float16")

SIZE = 160
base_model, model = get_model(SIZE)
model.load_weights(f"model/model{SIZE}.ckpt")
test_ds, val_ds = get_dataset(False, SIZE, 1, 16, True)

conv_idx = [
    i for i, l in enumerate(base_model.layers) if isinstance(l, tf.keras.layers.Conv2D)
]

specs_list = prune_all(model, conv_idx, test_ds, val_ds)
with open("specs_list.pickle", "wb") as h:
    pickle.dump(specs_list, h)
