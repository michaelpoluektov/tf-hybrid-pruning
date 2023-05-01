from utils import (
    Eval,
    FixedLoss,
    FixedParams,
    find_compression_loss,
    find_compression_params,
    find_rank_loss,
    find_spar_loss,
)
import tensorflow.keras as K
import numpy as np
from typing import Callable, TypeVar

LP = TypeVar("LP", FixedLoss, FixedParams)
IF = TypeVar("IF", int, float)
COMP = TypeVar("COMP", IF, tuple[int, float])


def find_factors(
    model: K.Model,
    layers: list[K.layers.Layer],
    eval: Eval,
    fs: LP,
    func: Callable[[K.layers.Layer, Eval, LP], tuple[np.array, COMP]],
) -> dict[K.layers.Layer, COMP]:
    comp_dict = {}
    for l in layers:
        best_w, comp = func(l, eval, fs)
        l.set_weights([best_w, l.bias.numpy()])
        _, acc = eval.model.evaluate(eval.ds)
        eval.base_accuracy = acc
        comp_dict[l] = comp
        eval.pbar.update(1)
    return comp_dict


def find_factors_loss(
    model: K.Model, layers: list[K.layers.Layer], eval: Eval, fl: FixedLoss
) -> dict[K.layers.Layer, tuple[int, float]]:
    return find_factors(model, layers, eval, fl, find_compression_loss)


def find_factors_params(
    model: K.Model, layers: list[K.layers.Layer], eval: Eval, fp: FixedParams
) -> dict[K.layers.Layer, tuple[int, float]]:
    return find_factors(model, layers, eval, fp, find_compression_params)


def find_single_loss(
    model: K.Model,
    layers: list[K.layers.Layer],
    eval: Eval,
    fl: FixedLoss,
    mode="tucker",
) -> dict[K.layers.Layer, IF]:
    if mode not in ["tucker", "spar"]:
        raise ValueError(f"Mode most be either 'tucker' or 'spar', got {mode}")
    func = find_rank_loss if mode == "tucker" else find_spar_loss
    return find_factors(model, layers, eval, fl, func)
