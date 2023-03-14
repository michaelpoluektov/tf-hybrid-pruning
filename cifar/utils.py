import os
from typing import Callable
from dataclasses import dataclass
from sparse_conv2d import SparseConv2D

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import global_policy
from tqdm import tqdm


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


def copy_model(model):
    copy_model = tf.keras.models.clone_model(model)
    copy_model.set_weights(model.get_weights())
    return copy_model


def expand_layer(layer):
    if not isinstance(layer, SparseConv2D):
        raise AttributeError("Input must be a SparseConv2D layer.")
    l.in_mask = tf.ones(l.input_shape[1:], dtype=global_policy().compute_dtype)
    l.out_mask = tf.ones(l.output_shape[1:], dtype=global_policy().compute_dtype)


def shrink_layer(layer):
    if not isinstance(layer, SparseConv2D):
        raise AttributeError("Input must be a SparseConv2D layer.")
    l.in_mask = 1
    l.out_mask = 1


def copy_layer(layer) -> tf.keras.layers.Layer:
    config = layer.get_config()
    weights = layer.get_weights()
    layer2 = type(layer).from_config(config)
    layer2.build(layer.input_shape)
    layer2.set_weigthts(weights)
    return layer2


def get_pruned_accuracy(model, layer_id, threshold, test_ds):
    model_2 = copy_model(model)


def propagate_constants(model, layer_id, input_constants):
    if model.layers[layer_id].bias:
        layer2 = copy_layer(model.layers[layer_id])
        layer2.bias = None
        outputs = layer2(input_constants)
        model.layers[layer_id].bias += outputs
    else:
        raise NotImplementedError("Haven't done it for layer without bias yet.")


def prune_layer(model, layer_id, max_loss, pbar, test_ds):
    test_model = copy_model(model)


# def search_factors(
#    model: tf.keras.Model,
#    layer_id: int,
#    specs: LayerSpecs,
#    func: Callable,
#    test_ds: tf.data.Dataset,
#    importance_t: tf.Tensor,
#    max_loss: float,
#    pbar: tqdm,
# ) -> LayerSpecs:
#    base_model = model.layers[1]
#    t_min, t_max, t_cur = 0, 100, 50
#    while t_max - t_min > 100 / 2**6:
#        cutoff = tfp.stats.percentile(importance_t, t_cur)
#        if not cutoff == 0:
#            set_const(base_model.layers[layer_id], importance_t < cutoff, s.weights, func)
#            model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
#            _, out_accuracy = model.evaluate(test_ds, verbose=0)
#        else:
#            out_accuracy = specs.base_accuracy
#        score = get_score(out_accuracy - specs.base_accuracy, t_cur / 100, max_loss)
#        pbar.write(
#            f"Accuracy: {out_accuracy:.5f}, threshold: {t_cur}%, score: {score:.5f}",
#            end=" ",
#        )
#        if score >= 0:
#            if score > specs.best_score:
#                specs.best_score = score
#                specs.best_cutoff = cutoff
#                specs.best_sp = t_cur
#            t_min = t_cur
#            specs.min_cutoff = cutoff
#            pbar.write("(success)")
#        else:
#            t_max = t_cur
#            pbar.write("(fail)")
#        t_cur = (t_max - t_min) / 2 + t_min
#    return specs


# def prune_all(
#    model: tf.keras.Model,
#    conv_idx: list[int],
#    test_ds: tf.data.Dataset,
#    val_ds: tf.data.Dataset,
#    max_loss: float = 0.2,
# ):
#    # Assuming model is trained via transfer learning - base_model would be the original headless model
#    base_model = model.layers[1]
#    out_sparsities = []
#    losses_per_layer = get_weights(conv_idx, base_model) * max_loss
#    pbar = tqdm(total=len(losses_per_layer))
#    specs_list = []
#    for l, i in zip(losses_per_layer, conv_idx):
#        m = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[i].output)
#        activations = get_activations(m, 50, test_ds)
#        means = tf.math.reduce_mean(activations, 0)
#        abs_means = tf.math.reduce_mean(tf.math.abs(activations), 0)
#        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
#        s = LayerSpecs(base_accuracy=model.evaluate(test_ds, verbose=0, bias=base_model.layers[i].bias)[1])
#        pbar.write(f"BASE = {s.base_accuracy:.5f}, maximum accuracy loss: {l:.5f}")
#        # prune activations
#        s = search_factors(model, i, s, prune_out, test_ds, abs_means, l, pbar)
#        # set unpruned activations to mean
#
#        # prune in activations
#
#        # set unpruned in activations to mean then propagate up to the mask?
#
#        # prune weights
#
#        pbar.write(
#            f"Best score (activation pruning): {s.best_score:.5f} ({s.best_sp}%)"
#        )
#        pbar.update(1)
#        set_const(base_model.layers[i], abs_means < s.best_cutoff, None, prune_out)
#        specs_list.append(s)
#    return specs_list
