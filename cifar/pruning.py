import tensorflow as tf
import numpy as np
from model import get_resnet
from dataset import get_dataset
from tensorflow.keras import mixed_precision
from utils import sparsity_only, decomp_only, Eval, compress_and_val
from tqdm import tqdm
import pickle

mixed_precision.set_global_policy("mixed_float16")
names = ["resnet_l1", "resnet_l2", "resnet_noreg", "resnet_bn_finetune"]
funcs = [compress_and_val]
# funcs = [sparsity_only, decomp_only]

for name in names:
    for f in funcs:
        SIZE = 224
        base_model, model = get_resnet()
        test_ds, val_ds = get_dataset(False, SIZE, 1, 16, True)
        model.load_weights(f"model/{name}.h5")
        conv_idx = [
            i
            for i, l in enumerate(base_model.layers)
            if isinstance(l, tf.keras.layers.Conv2D) and l.kernel.shape[0] == 3
        ]
        props = []
        model.compile(metrics=["accuracy"])
        _, base_accuracy = model.evaluate(test_ds)
        e = Eval(model, test_ds, tqdm(total=len(conv_idx)), base_accuracy)
        for idx in conv_idx:
            l = base_model.layers[idx]
            prop, acc = f(l, e)
            props.append(prop)
            e.base_accuracy = acc
            e.pbar.update(1)
        model.evaluate(val_ds)

        print(f"MEAN: {np.mean(props)}, ARR: {props}")
        with open(f"model/{name}_{f.__name__}.pickle", "wb") as h:
            pickle.dump(props, h)
