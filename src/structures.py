from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from numpy import ndarray
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class Eval:
    model: Model
    ds: Dataset
    pbar: tqdm
    base_accuracy: float


def test_weights_eval(eval: Eval, l: Layer, new_w: ndarray, d_threshold: float = 0.001):
    default_w = l.kernel.numpy()
    l.set_weights([new_w, l.bias.numpy()])
    _, acc = eval.model.evaluate(eval.ds, verbose=0)
    l.set_weights([default_w, l.bias.numpy()])
    return acc >= eval.base_accuracy - d_threshold


@dataclass
class PruningStructure:
    transform_mask: Callable[
        [np.ndarray, tuple[int, int, int, int]], ndarray
    ] = lambda k, shape: k
    reduce_ker: Callable = lambda x: x


@dataclass
class FixedLoss:
    pruning_structure: PruningStructure = PruningStructure()
    decomp_weight_func: Callable[float, float] = lambda x: x
    spar_weight_func: Callable[float, float] = lambda x: 4 * x
    inv_spar_weight_func: Callable[float, float] = lambda x: x / 4
    eval_func: Callable[[Eval, Layer, ndarray], bool] = test_weights_eval


@dataclass
class FixedParams:
    weight_dict: dict[Layer, float]
    pruning_structure: PruningStructure = PruningStructure()
    decomp_weight_func: Callable[float, float] = lambda x: x
    spar_weight_func: Callable[float, float] = lambda x: x * 4
    inv_spar_weight_func: Callable[float, float] = lambda x: x / 4
    eval_func: Callable[[Eval, Layer, ndarray], float] = lambda e, l, k: np.mean(
        (l.kernel.numpy() - k) ** 2
    )


def get_channel_ps():
    def reduce_ker(k):
        return np.mean(k, axis=2)

    def transform_mask(mask, shape):
        mask_expanded = mask[:, :, np.newaxis, :]
        return np.broadcast_to(mask_expanded, shape)

    return PruningStructure(reduce_ker=reduce_ker, transform_mask=transform_mask)


def get_filter_ps():
    def reduce_ker(k):
        return np.mean(k, axis=3)

    def transform_mask(mask, shape):
        mask_expanded = mask[:, :, :, np.newaxis]
        return np.broadcast_to(mask_expanded, shape)

    return PruningStructure(reduce_ker=reduce_ker, transform_mask=transform_mask)


def get_block_ps(block_size):
    if 64 % block_size[0] != 0 or 64 % block_size[1] != 0:
        raise AttributeError(
            f"Invalid shape for block structure: must be a factor of 64, got {block_size}"
        )

    def get_k_shape(k_shape, block_size):
        return (
            k_shape[:2]
            + (k_shape[2] // block_size[0], block_size[0])
            + (k_shape[3] // block_size[1], block_size[1])
        )

    def reduce_ker(k):
        new_k_shape = get_k_shape(k.shape, block_size)
        new_k = k.reshape(new_k_shape)
        return np.mean(new_k, axis=(3, 5))

    def transform_mask(mask, shape):
        mask_expanded = mask[:, :, :, np.newaxis, :, np.newaxis]
        broadcast_shape = mask_expanded.shape[:3] + (
            block_size[0],
            mask_expanded.shape[4],
            block_size[1],
        )
        broadcast_mask = np.broadcast_to(mask_expanded, broadcast_shape)
        return broadcast_mask.reshape(shape)

    return PruningStructure(reduce_ker=reduce_ker, transform_mask=transform_mask)
