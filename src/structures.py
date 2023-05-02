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
