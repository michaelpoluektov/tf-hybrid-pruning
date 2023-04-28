import os
import scipy as sp
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import partial_tucker
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class Eval:
    model: tf.keras.Model
    ds: tf.data.Dataset
    pbar: tqdm
    base_accuracy: float


def get_spars(spar_limit=100):
    # 0 and powers of 2 until 25.6%
    arr = [0] + [0.1 * 1.3**i for i in range(19) if 0.1 * 1.3**i < spar_limit]
    return arr


def get_props(c):
    m = c // 16
    return [m * i for i in range(4, 13)]  # THIS WAS (1, 13)


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


def get_whatif(k, modes=(2, 3), rank=1, spar=90):
    if rank:
        core, first, last, sp = get_decomp(k, modes, rank, spar)
        b = tl.tenalg.multi_mode_dot(core, (first, last), modes=modes)
        return b + sp
    else:
        t = np.percentile(abs(k), spar)
        mask = abs(k) > t
        b = np.zeros(k.shape)
        b[mask] = k[mask]
        return b


def get_spr_weights(k, modes=(2, 3), rank=1, spar=90):
    core, first, last, _ = get_decomp(k, modes, rank, spar)
    b = tl.tenalg.multi_mode_dot(core, (first, last), modes=modes)
    return b


def test_sparsity(l, eval, spar, rank):
    default_w = l.kernel.numpy()
    b = l.bias.numpy()
    prop_w = get_whatif(default_w, (2, 3), rank, 100 - spar)
    l.set_weights([prop_w, b])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, b])
    return prop_w, acc


def sparsity_only(l, eval):
    for i in range(10, 100, 5):
        w, acc = test_sparsity(l, eval, i, 0)
        if acc >= eval.base_accuracy - 0.001:
            l.set_weights([w, l.bias.numpy()])
            return i / 100, acc
    return 1, eval.base_accuracy


def decomp_only(l, eval):
    c = l.kernel.shape[-1]
    for i in range(1, 14):
        r = c * i // 16
        w, acc = test_sparsity(l, eval, 0, [r, r])
        if acc >= eval.base_accuracy - 0.001:
            l.set_weights([w, l.bias.numpy()])
            return get_decomp_weight(l, [r, r]) / get_weight(l), acc
    return 1, eval.base_accuracy


def try_sparsities(l, eval, default_w, rank):
    spars = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.8]
    lst = []
    for s in spars:
        new_w, acc = test_sparsity(l, eval, s, rank)
        diff = (default_w - new_w).flatten()
        std = diff.std()
        mean_diff = np.mean(diff**2)
        kurt = sp.stats.kurtosis(diff)
        spread = np.max(diff) - np.min(diff)
        lst.append((std, mean_diff, kurt, spread, acc))
        print(acc)
        eval.pbar.update(1)
    return lst


def get_decomp_stats(l, eval: Eval):
    default_w = l.kernel.numpy()
    c = default_w.shape[-1]
    props = [i for i in range((c * 3) // 16, c, c // 16)]
    llst = []
    for i, prop in enumerate(props):
        eval.pbar.write(f"Testing prop: {prop}...", end=" ")
        rank = [prop, prop]
        lst = try_sparsities(l, eval, default_w, rank)
        llst.append(lst)
    return llst


def find_sparsity(l: tf.keras.layers.Layer, eval: Eval, default_w, spar_limit, rank):
    spars = get_spars(spar_limit * 100)
    min_spar, max_spar = 0.0, 100.0
    best_w = default_w
    while spars:
        spar = spars[len(spars) // 2]
        new_w, acc = test_sparsity(l, eval, spar, rank)
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
        new_w, acc = test_sparsity(l, eval, spar, rank)
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
    default_w = l.kernel.numpy()
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
        spar_w = get_weight(l) * spar * 4 / 100
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


def compress_and_val(l, eval: Eval):
    best_w, best_pair = find_compression(l, eval)
    l.set_weights([best_w, l.bias.numpy()])
    _, acc = eval.model.evaluate(eval.ds)
    eval.base_accuracy = acc
    return best_pair, acc


def get_weight(layer: tf.keras.layers.Conv2D) -> float:
    return np.prod(np.array(layer.kernel.shape)) * np.prod(layer.output_shape[1:-1])


def get_weights(conv_idx: list[int], model: tf.keras.Model) -> np.array:
    weight_arr = np.array([get_weight(model.layers[i]) for i in conv_idx])
    return weight_arr / np.sum(weight_arr)


def copy_model(model):
    copy_model = tf.keras.models.clone_model(model)
    copy_model.set_weights(model.get_weights())
    return copy_model
