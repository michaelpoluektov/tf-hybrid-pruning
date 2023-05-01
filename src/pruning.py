from utils import (
    Eval,
    FixedLoss,
    FixedParams,
    find_compression_loss,
    find_compression_params,
)
import tensorflow.keras as K


def find_factors_loss(
    model: K.Model, layers: list[K.layers.Layer], eval: Eval, fl: FixedLoss
):
    compression_pairs = {}
    for l in layers:
        best_w, best_pair = find_compression_loss(l, eval, fl)
        l.set_weights([best_w, l.bias.numpy()])
        _, acc = eval.model.evaluate(eval.ds)
        eval.base_accuracy = acc
        compression_pairs[l] = best_pair
        eval.pbar.update(1)
    return compression_pairs


def find_factors_params(
    model: K.Model, layers: list[K.layers.Layer], eval: Eval, fp: FixedParams
):
    compression_pairs = {}
    for l in layers:
        _, best_pair = find_compression_params(l, eval, fp)
        compression_pairs[l] = best_pair
        eval.pbar.update(1)
    return compression_pairs
