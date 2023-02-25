import os
from typing import Callable
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm


@dataclass
class LayerSpecs:
    base_accuracy: float
    best_score: int = 0
    best_sp: int = 0
    min_cutoff: int = 0


def apply_mask(out: tf.Tensor, mask: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return out * tf.cast(tf.logical_not(mask), tf.float16) + weights * tf.cast(
        mask, tf.float32
    )


def prune_out(out: tf.Tensor, mask: tf.Tensor, weights: tf.Tensor = None) -> tf.Tensor:
    return out * tf.cast(tf.logical_not(mask), tf.float16)


def call_wrapper(
    func: Callable, mask: tf.Tensor, weights: tf.Tensor, pruning_func: Callable
) -> Callable:
    def wrapper(*args, **kwargs):
        return pruning_func(func(*args, **kwargs), mask, weights)

    return wrapper


def set_const(
    layer: tf.keras.layers.Layer,
    mask: tf.Tensor,
    weights: tf.Tensor,
    pruning_func: Callable,
) -> None:
    if not "partial" in layer.name:
        layer.default_call = layer.call
        layer._name = f"partial_{layer.name}"
    layer._mask = mask
    layer.call = call_wrapper(layer.default_call, layer._mask, weights, pruning_func)


def get_conv_idx_out(model: tf.keras.Model) -> list[int]:
    return [
        i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.Conv2D)
    ]


def get_conv_idx_in(model: tf.keras.Model) -> list[int]:
    layers = []
    for layer in model.layers:
        outs = [n.outbound_layer for n in layer._outbound_nodes]
        for out in outs:
            if isinstance(out, tf.keras.layers.Conv2D):
                layers.append(model.layers.index(out))
    return layers


def get_activations(
    model: tf.keras.Model, num_samples: int, ds: tf.data.Dataset
) -> tf.Tensor:
    activations = []
    for j, (d, t) in enumerate(ds):
        if j == num_samples:
            break
        activations.append(model(d))
    return tf.concat(activations, 0)


def get_weight(layer: tf.keras.layers.Conv2D) -> float:
    return np.prod(np.array(layer.kernel.shape)) * np.prod(layer.output_shape[1:-1])


def get_weights(conv_idx: list[int], model: tf.keras.Model) -> np.array:
    weight_arr = np.array([get_weight(model.layers[i]) for i in conv_idx])
    return weight_arr / np.sum(weight_arr)


def get_score(accuracy_delta: float, sparsity: float, factor: float) -> float:
    return accuracy_delta + sparsity * factor


def search_factors(
    model: tf.keras.Model,
    layer_id: int,
    specs: LayerSpecs,
    func: Callable,
    test_ds: tf.data.Dataset,
    importance_t: tf.Tensor,
    mean_t: tf.Tensor,
    max_loss: float,
    pbar: tqdm,
) -> LayerSpecs:
    base_model = model.layers[1]
    t_min, t_max, t_cur = 0, 100, 50
    while t_max - t_min > 100 / 2**6:
        cutoff = tfp.stats.percentile(importance_t, t_cur)
        if not cutoff == 0:
            set_const(base_model.layers[layer_id], importance_t < cutoff, None, func)
            model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
            _, out_accuracy = model.evaluate(test_ds, verbose=0)
        else:
            out_accuracy = specs.base_accuracy
        score = get_score(out_accuracy - specs.base_accuracy, t_cur / 100, max_loss)
        pbar.write(
            f"Accuracy: {out_accuracy:.5f}, threshold: {t_cur}%, score: {score:.5f}",
            end=" ",
        )
        if score >= 0:
            if score > specs.best_score:
                specs.best_score = score
                specs.best_sp = t_cur
            t_min = t_cur
            specs.min_cutoff = cutoff
            pbar.write("(success)")
        else:
            t_max = t_cur
            pbar.write("(fail)")
        t_cur = (t_max - t_min) / 2 + t_min
    return specs


def prune_all(
    model: tf.keras.Model,
    conv_idx: list[int],
    test_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    max_loss: float = 0.2,
):
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
        s = LayerSpecs(base_accuracy=model.evaluate(test_ds, verbose=0)[1])
        pbar.write(f"BASE = {s.base_accuracy:.5f}, maximum accuracy loss: {l:.5f}")
        # prune activations
        s = search_factors(model, i, s, prune_out, test_ds, abs_means, means, l, pbar)
        pbar.write(
            f"Best score (activation pruning): {s.best_score:.5f} ({s.best_sp}%)"
        )
        pbar.update(1)
        set_const(base_model.layers[i], abs_means < s.best_sp, None, prune_out)
