import os
import scipy as sp
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import partial_tucker
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable
from tensorflow.keras.layers import Layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class Eval:
    model: tf.keras.Model
    ds: tf.data.Dataset
    pbar: tqdm
    base_accuracy: float


def test_weights_eval(
    eval: Eval, l: Layer, new_w: np.ndarray, d_threshold: float = 0.001
):
    default_w = l.kernel.numpy()
    l.set_weights([new_w, l.bias.numpy()])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, l.bias.numpy()])
    return acc >= eval.base_accuracy - d_threshold


@dataclass
class PruningStructure:
    transform_mask: Callable[
        [np.ndarray, tuple[int, int, int, int]], np.ndarray
    ] = lambda k, shape: k
    reduce_ker: Callable = lambda x: x


@dataclass
class FixedLoss:
    pruning_structure: PruningStructure = PruningStructure()
    decomp_weight_func: Callable[float, float] = lambda x: x
    spar_weight_func: Callable[float, float] = lambda x: 4 * x
    inv_spar_weight_func: Callable[float, float] = lambda x: x / 4
    eval_func: Callable[[Eval, Layer, np.ndarray], bool] = test_weights_eval


@dataclass
class FixedParams:
    weight_dict: dict[Layer, float]
    pruning_structure: PruningStructure = PruningStructure()
    decomp_weight_func: Callable[float, float] = lambda x: x
    spar_weight_func: Callable[float, float] = lambda x: x * 4
    inv_spar_weight_func: Callable[float, float] = lambda x: x / 4
    eval_func: Callable[[Eval, Layer, np.ndarray], float] = lambda e, l, k: np.mean(
        (l.kernel.numpy() - k) ** 2
    )


def get_spars(spar_limit: int = 100):
    # 0 and powers of 2 until 25.6%
    arr = [0] + [0.1 * 1.3**i for i in range(19) if 0.1 * 1.3**i < spar_limit]
    return arr


def get_props(c: int):
    m = c // 16
    return [m * i for i in range(4, 15)]  # THIS WAS (1, 13)


def get_decomp(
    k: np.ndarray,
    structure: PruningStructure,
    modes: tuple[int, int] = (2, 3),
    rank: int = 1,
    spar: float = 90.0,
):
    rank = [rank, rank]
    (core, factors), _ = partial_tucker(k, modes=modes, rank=rank)
    b = tl.tenalg.multi_mode_dot(core, factors, modes)
    for _ in range(5):
        diff = structure.reduce_ker(abs(k - b))
        t = np.percentile(diff, spar)
        mask = structure.transform_mask(diff >= t, k.shape)
        c = k.copy()
        c[mask] = b[mask]
        (core, factors), _ = partial_tucker(c, modes=modes, rank=rank)
        b = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    sp = np.zeros(k.shape)
    sp[mask] = k[mask] - b[mask]
    return core, *factors, sp.astype(np.float32)


def get_whatif(
    k: np.ndarray,
    structure: PruningStructure,
    modes: tuple[int, int] = (2, 3),
    rank: int = 1,
    spar: float = 90.0,
):
    if rank:
        core, first, last, sp = get_decomp(k, structure, modes, rank, spar)
        b = tl.tenalg.multi_mode_dot(core, (first, last), modes=modes)
        return b + sp
    else:
        t = np.percentile(structure.reduce_ker(abs(k)), spar)
        mask = structure.transform_mask(abs(k) > t, k.shape)
        b = np.zeros(k.shape)
        b[mask] = k[mask]
        return b


def find_sparsity(
    l: Layer,
    eval: Eval,
    default_w: np.ndarray,
    spar_limit: float,
    rank: int,
    fl: FixedLoss,
):
    spars = get_spars(spar_limit * 100)
    max_spar = 100.0
    best_w = default_w
    # tested_max = False
    while spars:
        # if not tested_max:
        #     spar = spars[-1]
        #     tested_max = True
        # else:
        spar = spars[len(spars) // 2]
        new_w = get_whatif(
            l.kernel.numpy(), fl.pruning_structure, (2, 3), rank, 100 - spar
        )
        acceptable = fl.eval_func(eval, l, new_w)
        if acceptable:
            max_spar = spar
            best_w = new_w
            spars = list(filter(lambda e: e < spar, spars))
        else:
            spars = list(filter(lambda e: e > spar, spars))
    if max_spar == 100:
        eval.pbar.write("Bad, returning...")
        return default_w, 100.0
    elif max_spar == 0.0:
        eval.pbar.write("No need for sparsity...")
        return best_w, 0.0
    else:
        eval.pbar.write(f"Found sparsity: {max_spar:.1f}%")
        return best_w, max_spar


def find_compression_loss(l: Layer, eval: Eval, fl: FixedLoss):
    default_w = l.kernel.numpy()
    best_w, best_score, best_pair = default_w, 1, (0, 100)
    ranks = get_props(default_w.shape[-1])
    for i, rank in enumerate(ranks):
        eval.pbar.write(f"Testing Rank: {rank}...", end=" ")
        prop_w = fl.decomp_weight_func(get_decomp_weight(l, rank) / get_weight(l))
        if prop_w > best_score:
            eval.pbar.write("Breaking.")
            break
        spar_limit = fl.inv_spar_weight_func(best_score - prop_w)
        new_w, spar = find_sparsity(l, eval, default_w, spar_limit, rank, fl)
        spar_w = fl.spar_weight_func(spar / 100)
        tot_score = prop_w + spar_w
        if tot_score <= best_score:
            best_score = tot_score
            best_w = new_w
            best_pair = (rank, spar)
    eval.pbar.write(
        f"Best pair: decomposition = {best_pair[0]},"
        + f" sparsity = {best_pair[1]:4f}, score: {best_score:.4f}"
    )
    return best_w, best_pair


def find_compression_params(
    l: Layer, eval: Eval, fp: FixedParams, use_optimisation: bool = False
) -> tuple[np.ndarray, tuple[int, float]]:
    default_w = l.kernel.numpy()
    ranks = get_props(default_w.shape[-1])[:: (-1 if use_optimisation else 1)]
    min_loss = 1e31
    best_pair = 0, 100
    best_w = default_w
    for i, rank in enumerate(ranks):
        prop_w = fp.decomp_weight_func(get_decomp_weight(l, rank) / get_weight(l))
        if prop_w > fp.weight_dict[l]:
            if use_optimisation:
                continue
            else:
                break
        eval.pbar.write(f"Testing Rank: {rank}...", end=" ")
        spar = fp.inv_spar_weight_func(fp.weight_dict[l] - prop_w) * 100
        new_k = get_whatif(default_w, fp.pruning_structure, rank=rank, spar=100 - spar)
        loss = fp.eval_func(eval, l, new_k)
        if loss < min_loss:
            min_loss = loss
            best_pair = rank, spar
            best_w = new_k
            eval.pbar.write(f"Yes, sparsity: {spar:.2f}.")
        else:
            eval.pbar.write("No.")
            if use_optimisation:
                break
    eval.pbar.write(
        f"Best pair: decomposition = {best_pair[0]}, sparsity = {best_pair[1]:2f}%"
    )
    return best_w, best_pair


def find_rank_loss(l: Layer, eval: Eval, fl: FixedLoss) -> tuple[np.ndarray, int]:
    k = l.kernel.numpy()
    c = k.shape[-1]
    ranks = get_props(c)
    min_idx, max_idx = 0, len(ranks) - 1
    best_w, best_rank = k, 0
    while max_idx - min_idx > 1:
        cur_idx = (max_idx - min_idx) // 2
        rank = ranks[cur_idx]
        new_w = get_compressed_weights(l, rank=rank)
        if fl.eval_func(eval, l, new_w):
            best_w, best_rank = k, rank
            max_idx = cur_idx
        else:
            min_idx = cur_idx
    return best_w, best_rank


def find_spar_loss(l: Layer, eval: Eval, fl: FixedLoss) -> tuple[np.ndarray, float]:
    k = l.kernel.numpy()
    spars = [i for i in range(1, 100) if i < fl.inv_spar_weight_func(100)]
    min_idx, max_idx = 0, len(spars) - 1
    best_w, best_spar = k, 100
    while max_idx - min_idx > 1:
        cur_idx = (max_idx - min_idx) // 2
        spar = spars[cur_idx]
        new_w = get_whatif(k, fl.pruning_structure, rank=0, spar=spar)
        if fl.eval_func(eval, l, new_w):
            best_w, best_spar = k, spar
            max_idx = cur_idx
        else:
            min_idx = cur_idx
    return best_w, best_spar


def get_weight(layer: tf.keras.layers.Conv2D) -> float:
    return np.prod(np.array(layer.kernel.shape)) * np.prod(layer.output_shape[1:-1])


def get_weights(conv_idx: list[int], model: tf.keras.Model) -> np.ndarray:
    weight_arr = np.array([get_weight(model.layers[i]) for i in conv_idx])
    return weight_arr / np.sum(weight_arr)


def copy_model(model):
    copy_model = tf.keras.models.clone_model(model)
    copy_model.set_weights(model.get_weights())
    return copy_model


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


def get_decomp_weight(l, rank):
    rank = [rank, rank]
    sz = np.prod(l.output_shape[1:-1])
    kshape = l.kernel.shape
    return sz * (
        np.prod(kshape[:2]) * np.prod(rank)
        + rank[0] * kshape[-2]
        + rank[1] * kshape[-1]
    )


def sparsity_only(l, eval):
    for i in range(10, 100, 5):
        w, acc = test_sparsity(l, eval, i, 0)
        if acc >= eval.base_accuracy - 0.001:
            l.set_weights([w, l.bias.numpy()])
            return i / 100, acc
    return 1, eval.base_accuracy


def decomp_only_stats(l, eval):
    c = l.kernel.shape[-1]
    for i in range(1, 14):
        r = c * i // 16
        w, acc = test_sparsity(l, eval, 0, [r, r])
        if acc >= eval.base_accuracy - 0.001:
            l.set_weights([w, l.bias.numpy()])
            return get_decomp_weight(l, [r, r]) / get_weight(l), acc
    return 1, eval.base_accuracy


def get_spr_weights(k, modes=(2, 3), rank=1, spar=90):
    core, first, last, _ = get_decomp(k, None, modes, rank, spar)
    b = tl.tenalg.multi_mode_dot(core, (first, last), modes=modes)
    return b


def get_compressed_weights(layer, modes=(2, 3), rank=1) -> tuple[np.ndarray, int]:
    if len(modes) != 2 and len(modes) != 1:
        raise Exception(f"Modes doesn't make sense: {modes}")
    (core, factors), _ = partial_tucker(
        layer.kernel.numpy(), modes=modes, rank=rank, init="svd"
    )
    new_w = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    return new_w


def test_sparsity(l, eval, spar, rank):
    default_w = l.kernel.numpy()
    b = l.bias.numpy()
    prop_w = get_whatif(default_w, (2, 3), rank, 100 - spar)
    l.set_weights([prop_w, b])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, b])
    return prop_w, acc
