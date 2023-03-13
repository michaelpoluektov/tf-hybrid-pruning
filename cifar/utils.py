import os
from typing import Callable
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Conv2D


@register_keras_serializable()
class SparseConv2D(tf.keras.layers.Conv2D):
    def __init__(self, in_mask=1, out_mask=1, in_threshold=0, **kwargs):
        super().__init__(**kwargs)
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_threshold = in_threshold

    def call(self, inputs):
        masked_inputs = (
            tf.dtypes.cast(
                tf.math.abs(inputs) > self.in_threshold,
                tf.keras.mixed_precision.global_policy().compute_dtype,
            )
            * inputs
            * self.in_mask
        )
        convolved = super().call(masked_inputs)
        masked_outputs = convolved * self.out_mask
        return masked_outputs

    def get_config(self):
        config = super().get_config()
        config["in_mask"] = self.in_mask
        config["out_mask"] = self.out_mask
        config["in_threshold"] = self.in_threshold
        return config

    @classmethod
    def from_config(cls, config):
        if "in_mask" in config and "out_mask" in config and "in_threshold" in config:
            in_mask = config["in_mask"]
            out_mask = config["out_mask"]
            in_threshold = config["in_threshold"]
            config.pop("in_mask")
            config.pop("out_mask")
            config.pop("in_threshold")
        else:
            in_mask = 1
            out_mask = 1
            in_threshold = 0
        layer = cls(
            in_mask=in_mask, out_mask=out_mask, in_threshold=in_threshold, **config
        )
        return layer


def clone_function(layer):
    if isinstance(layer, Conv2D):
        return SparseConv2D.from_config(layer.get_config())
    else:
        return layer.__class__.from_config(layer.get_config())


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
        return layer2(input_constants)


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
