import os
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import partial_tucker
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import global_policy
from tensorflow.keras.layers import (
    Multiply,
    Add,
    Dropout,
    Activation,
    BatchNormalization,
)
from tqdm import tqdm
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class Eval:
    model: tf.keras.Model
    ds: tf.data.Dataset
    pbar: tqdm
    base_accuracy: float


def get_layer_model(layer, rank=1):
    (core, (first, last)), rec_errors = partial_tucker(
        layer.kernel.numpy(), modes=[2, 3], rank=rank, init="svd", n_iter_max=1000
    )
    inp = tf.keras.layers.Input(shape=layer.input_shape[1:])
    l1 = tf.keras.layers.Conv2D(
        filters=first.shape[1],
        kernel_size=(1, 1),
        use_bias=False,
        dilation_rate=layer.dilation_rate,
        kernel_initializer=(
            lambda x, dtype=None: tf.convert_to_tensor(np.expand_dims(first, [0, 1]))
        ),
    )(inp)
    l2 = tf.keras.layers.Conv2D(
        filters=core.shape[-1],
        kernel_size=layer.kernel_size,
        use_bias=False,
        dilation_rate=layer.dilation_rate,
        strides=layer.strides,
        padding=layer.padding,
        kernel_initializer=(lambda x, dtype=None: tf.convert_to_tensor(core)),
    )(l1)
    l3 = tf.keras.layers.Conv2D(
        filters=last.shape[0],
        kernel_size=(1, 1),
        use_bias=True,
        dilation_rate=layer.dilation_rate,
        activation=layer.activation,
        kernel_initializer=(
            lambda x, dtype=None: tf.convert_to_tensor(
                np.expand_dims(last.T, [0, 1]), layer.get_weights()[1]
            )
        ),
        bias_initializer=(
            lambda x, dtype=None: tf.convert_to_tensor(layer.get_weights()[1])
        ),
    )(l2)
    return tf.keras.Model(inputs=inp, outputs=l3)


def get_spars(spar_limit=100):
    # 0 and powers of 2 until 25.6%
    arr = (
        [0]
        + [0.1 * 1.3**i for i in range(22) if 0.1 * 1.5**i < spar_limit]
        + [spar_limit]
    )
    return arr


def get_props(c):
    # for 64: [0, 1, 2, 4, 8, 16, 32, 48]
    return [0] + [2**i for i in range(int(np.log2(c)))] + [3 * c // 4]


def get_compressed_weights(layer, modes=(2, 3), rank=1) -> tuple[np.array, int]:
    if len(modes) != 2 and len(modes) != 1:
        raise Exception(f"Modes doesn't make sense: {modes}")
    (core, factors), _ = partial_tucker(
        layer.kernel.numpy(), modes=modes, rank=rank, init="svd"
    )
    sz = np.prod(layer.output_shape[1:-1])
    new_w = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    if len(modes) == 2:
        first, last = factors
        w1 = sz * np.prod(first.shape)
        w2 = sz * np.prod(core.shape)
        w3 = sz * np.prod(last.shape)
        tot_w = w1 + w2 + w3
    else:
        tot_w = sz * (np.prod(core.shape) + np.prod(factors[0].shape))
    return new_w, tot_w


def test_sparsity(l, eval, default_w, prop_w, b, spar):
    t = np.percentile(abs(prop_w - default_w), 100 - spar)
    mask = abs(prop_w - default_w) > t
    s_w = prop_w[:]
    s_w[mask] = default_w[mask]
    l.set_weights([s_w, b])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([prop_w, b])
    return s_w, acc


def find_sparsity(l: tf.keras.layers.Layer, eval: Eval, default_w, spar_limit):
    spars = get_spars(spar_limit * 100)
    prop_w, b = l.kernel.numpy(), l.bias.numpy()
    min_spar, max_spar = 0.0, 100.0
    best_w = default_w
    while spars:
        spar = spars[len(spars) // 2]
        new_w, acc = test_sparsity(l, eval, default_w, prop_w, b, spar)
        if acc >= eval.base_accuracy:
            max_spar = spar
            best_w = new_w
            spars = list(filter(lambda e: e < spar, spars))
        elif acc < eval.base_accuracy - 0.03:
            min_spar = spar
            spars = list(filter(lambda e: e > spar, spars))
        else:
            spars.remove(spar)
    if max_spar == 100 and min_spar == spar_limit * 100:
        eval.pbar.write("everything sucks, returning...", end=" ")
        return default_w, 100.0
    if max_spar == 0.0:
        eval.pbar.write("No need for sparsity...", end=" ")
        return best_w, 0.0
    if max_spar == 100:
        max_spar = spar_limit * 100
    eval.pbar.write(f"found min = {min_spar:.4f}%, max = {max_spar:.4f}%...", end=" ")
    incr = (max_spar - min_spar) / 10
    spars = np.arange(min_spar + incr, max_spar + incr / 2, incr)
    for spar in spars:
        new_w, acc = test_sparsity(l, eval, default_w, prop_w, b, spar)
        if acc >= eval.base_accuracy:
            return new_w, spar
    eval.pbar.write("everything sucks, returning...", end=" ")
    return default_w, 100.0


def get_decomp_weight(l, rank):
    sz = np.prod(l.output_shape[1:-1])
    kshape = l.kernel.shape
    return sz * (
        np.prod(kshape[:2]) * np.prod(rank)
        + rank[0] * kshape[-2]
        + rank[1] * kshape[-1]
    )


def find_compression(l, eval: Eval):
    default_w, default_b = l.kernel.numpy(), l.bias.numpy()
    best_w, best_score, best_pair = default_w, 1, (0, 100)
    props = get_props(default_w.shape[-1])
    for i, prop in enumerate(props):
        eval.pbar.write(f"Testing prop: {prop}...", end=" ")
        rank = [prop, prop]
        prop_w = get_decomp_weight(l, rank)
        if prop_w > best_score * get_weight(l):
            eval.pbar.write("Breaking.")
            break
        if prop != 0:
            new_w, prop_ops = get_compressed_weights(l, rank=rank)
        else:
            new_w, prop_ops = np.zeros(default_w.shape), 0
        l.set_weights([new_w, default_b])
        spar_limit = best_score - prop_w / get_weight(l)
        new_w, spar = find_sparsity(l, eval, default_w, spar_limit)
        l.set_weights([default_w, default_b])
        eval.pbar.write(f"final sparsity: {spar:.2f}%")
        spar_w = get_weight(l) * spar / 100
        tot_score = (prop_w + spar_w) / get_weight(l)
        if tot_score <= best_score:
            best_score = tot_score
            best_w = new_w
            best_pair = (prop, spar)
            if tot_score < 0.02:
                break
    eval.pbar.write(
        f"Best pair: decomposition = {best_pair[0]}, sparsity = {best_pair[1]:4f}, score: {best_score:.4f}"
    )
    return best_w, best_pair


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
    layer2.set_weights(weights)
    return layer2
