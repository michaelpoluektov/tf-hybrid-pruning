import tensorflow as tf
import numpy as np
from model import get_resnet
from dataset import get_dataset
from tensorflow.keras import mixed_precision
from utils import Eval, compress_and_val, PruningStructure
from tqdm import tqdm
import pickle
from tensorflow.keras.layers import Layer

mixed_precision.set_global_policy("mixed_float16")
names = ["resnet_l1", "resnet_l2", "resnet_noreg", "resnet_bn_finetune"]
funcs = [compress_and_val]
# funcs = [sparsity_only, decomp_only]


def test_weights_eval(eval: Eval, l: Layer, new_w: np.array, d_threshold: float = 0.0):
    default_w = l.kernel.numpy()
    l.set_weights([new_w, l.bias.numpy()])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, l.bias.numpy()])
    return acc >= eval.base_accuracy - d_threshold


def mse_eval(eval: Eval, l: Layer, new_w: np.array, d_threshold: float = 3e-5):
    return np.mean((l.kernel.numpy() - new_w) ** 2) < d_threshold


def kernel_similarity(k, k_decomp):
    k_normalized = k / np.linalg.norm(k)
    k_decomp_normalized = k_decomp / np.linalg.norm(k_decomp)
    return np.inner(k_normalized.ravel(), k_decomp_normalized.ravel())


def ks_eval(eval: Eval, l: Layer, new_w: np.array, d_threshold: float = 0.15):
    return kernel_similarity(l.kernel.numpy(), new_w) > 0.85


p1 = PruningStructure(eval_func=mse_eval)
p2 = PruningStructure(eval_func=lambda eval, l, new_w: mse_eval(eval, l, new_w, 5e-5))
p3 = PruningStructure(eval_func=ks_eval)
p4 = PruningStructure()
ps = [p1, p2, p3, p4]
func_names = ["mse1", "mse15", "ks", "acc"]
for name in names:
    for f in funcs:
        for p, fn in zip(ps, func_names):
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
                prop, acc = f(l, e, p)
                props.append(prop)
                e.base_accuracy = acc
                e.pbar.update(1)
            model.evaluate(val_ds)

            print(f"MEAN: {np.mean(props)}, ARR: {props}")
            with open(f"model/{name}_{f.__name__}_{fn}.pickle", "wb") as h:
                pickle.dump(props, h)
