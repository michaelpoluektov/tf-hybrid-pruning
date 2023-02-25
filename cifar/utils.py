import os
from dataclasses import dataclass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm


@dataclass
class LayerSpecs:
    base_accuracy: float
    t_min: int = 0
    t_cur: int = 50
    t_max: int = 100
    best_score: int = 0
    best_sp: int = 0
    min_cutoff: int = 0


def apply_mask(out, mask, weights):
    return out * tf.cast(tf.logical_not(mask), tf.float16) + weights * tf.cast(
        mask, tf.float32
    )


def prune_out(out, mask, weights=None):
    return out * tf.cast(tf.logical_not(mask), tf.float16)


def call_wrapper(func, mask, weights, pruning_func):
    def wrapper(*args, **kwargs):
        return pruning_func(func(*args, **kwargs), mask, weights)

    return wrapper


def set_const(layer, mask, weights, pruning_func):
    if not "partial" in layer.name:
        layer.default_call = layer.call
        layer._name = f"partial_{layer.name}"
    layer._mask = mask
    layer.call = call_wrapper(layer.default_call, layer._mask, weights, pruning_func)


def get_conv_idx_out(model):
    return [
        i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.Conv2D)
    ]


def get_conv_idx_in(model):
    layers = []
    for l in enumerate(model.layers):
        outs = [n.outbound_layer for n in layer._outbound_nodes]
        for out in outs:
            if isinstance(out, tf.keras.layers.Conv2D):
                layers.append(model.layers.index(out))
    return layers


def get_activations(model, num_samples, ds):
    activations = []
    for j, (d, t) in enumerate(ds):
        if j == num_samples:
            break
        activations.append(model(d))
    return tf.concat(activations, 0)


def get_weight(layer: tf.keras.layers.Conv2D):
    return np.prod(np.array(layer.kernel.shape)) * np.prod(layer.output_shape[1:-1])


def get_weights(conv_idx, model):
    weight_arr = np.array([get_weight(model.layers[i]) for i in conv_idx])
    return weight_arr / np.sum(weight_arr)


def get_score(accuracy_delta, sparsity, factor):
    return accuracy_delta + sparsity * factor


def prune_all(model, conv_idx, test_ds, val_ds, max_loss=0.2):
    # Assuming model is trained via transfer learning - base_model would be the original headless model
    base_model = model.layers[1]
    out_sparsities = []
    losses_per_layer = get_weights(conv_idx, base_model) * max_loss
    pbar = tqdm(total=len(losses_per_layer))
    for l, i in zip(losses_per_layer, conv_idx):
        m = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[i].output)
        activations = get_activations(m, 50, test_ds)
        means = tf.math.reduce_mean(activations, 0)
        abs_means = tf.math.reduce_mean(tf.math.abs(activations), 0)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        s = LayerSpecs(base_accuracy = model.evaluate(test_ds, verbose=0)[1])
        pbar.write(f"BASE = {s.base_accuracy:.5f}, maximum accuracy loss: {l:.5f}")
        while s.t_max - s.t_min > 100 / 2**6:
            cutoff = tfp.stats.percentile(abs_means, s.t_cur)
            if not cutoff == 0:
                set_const(base_model.layers[i], abs_means < cutoff, None, prune_out)
                model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
                _, out_accuracy = model.evaluate(test_ds, verbose=0)
            else:
                out_accuracy = s.base_accuracy
            score = get_score(out_accuracy-s.base_accuracy, s.t_cur/100, l)
            pbar.write(f"Accuracy: {out_accuracy:.5f}, threshold: {s.t_cur}%, score: {score:.5f}", end=" ")
            if score >= 0:
                if score > s.best_score:
                    s.best_score = score
                    s.best_sp = s.t_cur
                s.t_min = s.t_cur
                s.min_cutoff = cutoff
                pbar.write("(success)")
            else:
                s.t_max = s.t_cur
                pbar.write("(fail)")
            s.t_cur = (s.t_max - s.t_min) / 2 + s.t_min
        pbar.write(f"Best score: {s.best_score:.5f} ({s.best_sp}%)")
        pbar.update(1)
        set_const(base_model.layers[i], abs_means < s.best_sp, None, prune_out)
