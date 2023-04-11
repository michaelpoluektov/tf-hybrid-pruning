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


def get_layer_model(layer, core, first, last, sp_weights):
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
        kernel_initializer=(
            lambda x, dtype=None: tf.convert_to_tensor(
                np.expand_dims(last.T, [0, 1]), layer.get_weights()[1]
            )
        ),
    )(l2)
    # add sparse layer
    sp = tf.keras.layers.Conv2D(
        filters=last.shape[0],
        kernel_size=layer.kernel_size,
        use_bias=False,
        dilation_rate=layer.dilation_rate,
        strides=layer.strides,
        padding=layer.padding,
        kernel_initializer=(lambda x, dtype=None: tf.convert_to_tensor(sp_weights)),
    )(inp)
    outs = tf.keras.layers.Add()([l3, sp])
    # add bias
    b = layer.get_weights()[1].reshape(1, 1, 1, -1)
    outs = tf.keras.layers.Add()([outs, b])
    # add activation
    outs = layer.activation(outs)
    return tf.keras.Model(inputs=inp, outputs=outs)


def get_spars(spar_limit=100):
    # 0 and powers of 2 until 25.6%
    arr = [0] + [0.1 * 1.3**i for i in range(22) if 0.1 * 1.5**i < spar_limit]
    return arr


def get_props(c):
    m = c // 8
    return [m * i for i in range(1, 7)]


def get_compressed_weights(layer, modes=(2, 3), rank=1) -> tuple[np.array, int]:
    if len(modes) != 2 and len(modes) != 1:
        raise Exception(f"Modes doesn't make sense: {modes}")
    (core, factors), _ = partial_tucker(
        layer.kernel.numpy(), modes=modes, rank=rank, init="svd"
    )
    new_w = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    return new_w


def get_decomp(k, modes=(2, 3), rank=1, spar=90):
    (core, factors), _ = partial_tucker(k, modes=modes, rank=rank)
    b = tl.tenalg.multi_mode_dot(core, factors, modes)
    for _ in range(5):
        diff = abs(k - b)
        t = np.percentile(diff, spar)
        mask = diff >= t
        c = k.copy()
        c[mask] = b[mask]
        (core, factors), _ = partial_tucker(c, modes=modes, rank=rank)
        b = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    sp = np.zeros(k.shape)
    sp[mask] = k[mask] - b[mask]
    return core, *factors, sp.astype(np.float32)


def get_spr_weights(k, modes=(2, 3), rank=1, spar=90):
    core, first, last, _ = get_decomp(k, modes, rank, spar)
    b = tl.tenalg.multi_mode_dot(core, (first, last), modes=modes)
    return b


def test_sparsity(l, eval, b, spar, rank):
    default_w = l.kernel.numpy()
    prop_w = get_spr_weights(default_w, (2, 3), rank, 100 - spar)
    t = np.percentile(abs(prop_w - default_w), 100 - spar)
    mask = abs(prop_w - default_w) >= t
    prop_w[mask] = default_w[mask]
    l.set_weights([prop_w, b])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, b])
    return prop_w, acc


def find_sparsity(l: tf.keras.layers.Layer, eval: Eval, default_w, spar_limit, rank):
    spars = get_spars(spar_limit * 100)
    b = l.bias.numpy()
    min_spar, max_spar = 0.0, 100.0
    best_w = default_w
    while spars:
        spar = spars[len(spars) // 2]
        new_w, acc = test_sparsity(l, eval, b, spar, rank)
        if acc >= eval.base_accuracy - 0.002:
            max_spar = spar
            best_w = new_w
            spars = list(filter(lambda e: e < spar, spars))
        elif acc < eval.base_accuracy - 0.01:
            min_spar = spar
            spars = list(filter(lambda e: e > spar, spars))
        else:
            spars.remove(spar)
    if max_spar == 100 and min_spar == spar_limit * 100:
        eval.pbar.write("bad, returning...", end=" ")
        return default_w, 100.0
    if max_spar == 0.0:
        eval.pbar.write("No need for sparsity...", end=" ")
        return best_w, 0.0
    max_spar = min(spar_limit * 100, 25, max_spar)
    eval.pbar.write(f"found min = {min_spar:.2f}%, max = {max_spar:.2f}%...", end=" ")
    incr = (max_spar - min_spar) / 5
    spars = np.arange(min_spar + incr, max_spar + incr / 2, incr)
    for spar in spars:
        new_w, acc = test_sparsity(l, eval, b, spar, rank)
        if acc >= eval.base_accuracy - 0.002:
            return new_w, spar
    eval.pbar.write("bad, returning...", end=" ")
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
        spar_limit = best_score - prop_w / get_weight(l)
        new_w, spar = find_sparsity(l, eval, default_w, spar_limit, rank)
        eval.pbar.write(f"f. spar: {spar:.2f}%")
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
