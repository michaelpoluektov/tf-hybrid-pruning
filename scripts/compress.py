import sys
import os

sys.path.append("../src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pruning import find_factors_loss, find_factors_params
from model import get_resnet, get_decomp_resnet
from dataset import get_dataset
from utils import (
    get_decomp_weight,
    get_weight,
    find_rank_loss,
    find_spar_loss,
)
from structures import FixedLoss, FixedParams, Eval, test_weights_eval, PruningStructure
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from typing import Union, Callable, TypeVar

mixed_precision.set_global_policy("mixed_float16")


FP = TypeVar("FP", FixedLoss, FixedParams)


def tuple_of_ints(value):
    # Split the input string by comma and convert to a tuple of integers
    tuple_values = tuple(map(int, value.split(",")))

    if len(tuple_values) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected a tuple of 2 integers, but got {tuple_values}"
        )

    return tuple_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimise model and convert to SavedModel"
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input model weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the compressed SavedModel model",
    )
    parser.add_argument(
        "--pruning_structure",
        type=str,
        choices=["unstructured", "channel", "filter", "block"],
        default="unstructured",
        help="Pruning structure to use",
    )
    parser.add_argument(
        "--block_size",
        type=tuple_of_ints,
        help="Size of the block when using block pruning structure",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fixed_loss", "fixed_params"],
        default="fixed_loss",
        help="Optimization method to use: fixed_loss or fixed_params",
    )
    parser.add_argument(
        "--max_acc_loss",
        type=float,
        default=0.02,
        help="Maximum accuracy loss when using fixed_loss method",
    )
    parser.add_argument(
        "--compression_factor",
        type=float,
        help="Compression factor when using fixed_params method",
    )
    parser.add_argument(
        "--sparsity_weight",
        type=float,
        default=4.0,
        help="Relative contribution of the sparse Conv2D layer with respect to sparsity",
    )
    parser.add_argument(
        "--compression_type",
        type=str,
        choices=["hybrid", "tucker", "spars"],
        default="hybrid",
        help="Compression method",
    )
    args = parser.parse_args()
    if args.pruning_structure == "block" and args.block_size is None:
        parser.error("--block_size must be provided when using block pruning structure")
    if args.method == "fixed_loss" and args.max_acc_loss is None:
        parser.error("--max_acc_loss must be provided when using fixed_loss method")
    if args.method == "fixed_params" and args.compression_factor is None:
        parser.error(
            "--compression_factor must be provided when using fixed_params method"
        )
    if args.compression_factor is not None and (
        args.compression_factor <= 0 or args.compression_factor >= 1
    ):
        parser.error("Compression factor should be between 0 and 1 (exclusive)")
    args.input_path = os.path.abspath(args.input_path)
    args.output_path = os.path.abspath(args.output_path)
    if not os.path.exists(args.input_path):
        parser.error(f"Input path '{args.input_path}' does not exist")
    output_folder = os.path.dirname(args.output_path)
    if not os.path.exists(output_folder):
        parser.error(f"Output folder '{output_folder}' does not exist")
    return args


def get_pruning_structure(args):
    if args.pruning_structure == "unstructured":
        return PruningStructure()
    elif args.pruning_structure == "channel":

        def reduce_ker(k):
            return np.mean(k, axis=2)

        def transform_mask(mask, shape):
            mask_expanded = mask[:, :, np.newaxis, :]
            return np.broadcast_to(mask_expanded, shape)

        return PruningStructure(reduce_ker=reduce_ker, transform_mask=transform_mask)
    elif args.pruning_structure == "filter":

        def reduce_ker(k):
            return np.mean(k, axis=3)

        def transform_mask(mask, shape):
            mask_expanded = mask[:, :, :, np.newaxis]
            return np.broadcast_to(mask_expanded, shape)

        return PruningStructure(reduce_ker=reduce_ker, transform_mask=transform_mask)
    elif args.pruning_structure == "block":
        block_size = args.block_size
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

    else:
        raise NotImplementedError("block structure is not implemented yet")


def get_loss_eval_func(args):
    loss = args.max_acc_loss
    # CHANGE FOR 1x1
    per_layer_loss = loss / 16
    return lambda x, y, z: test_weights_eval(x, y, z, per_layer_loss)


def get_layers(base_model):
    return [
        l
        for l in base_model.layers
        if isinstance(l, tf.keras.layers.Conv2D) and l.kernel.shape[0] == 3
    ]


def compression(
    model: Model,
    ps: PruningStructure,
    base_model: Model,
    args: argparse.Namespace,
    fp: FP,
    comp_dict_func: Callable[
        [Model, list[Layer], Eval, FP], dict[Layer, tuple[int, float]]
    ],
):
    layers = get_layers(base_model)
    test_ds, val_ds = get_dataset(False, 224, 1, 16, True)
    print("Evaluating model...", end=" ")
    _, val_accuracy = model.evaluate(val_ds, verbose=0)
    _, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy: val={val_accuracy*100:.2f}%, test={test_accuracy*100:.2f}%")
    ev = Eval(
        model=model,
        ds=test_ds,
        pbar=tqdm(total=len(layers)),
        base_accuracy=test_accuracy,
    )
    comp_dict = comp_dict_func(base_model, layers, ev, fp)
    _, new_val = model.evaluate(val_ds, verbose=0)
    print(
        "Found factors. Accuracy loss: test"
        + f"={(test_accuracy - ev.base_accuracy)*100:.2}%, "
        + f"val={(val_accuracy-new_val)*100:.2f}%"
    )
    print("FACTORS:")
    for l in comp_dict:
        r, s = comp_dict[l]
        print(
            f"{l.name}: rank={r} "
            + f"({fp.decomp_weight_func(get_decomp_weight(l, r) / get_weight(l))*100:.2f}%), "
            + f"sparsity={s:.2f}% ({fp.spar_weight_func(s):.2f}%)"
        )
    pairs = [comp_dict[l] for l in layers]
    return pairs


def fixed_loss(model, ps, base_model, args):
    fl = FixedLoss(
        ps,
        eval_func=get_loss_eval_func(args),
        inv_spar_weight_func=lambda x: x / args.sparsity_weight,
        spar_weight_func=lambda x: args.sparsity_weight * x,
    )
    return compression(model, ps, base_model, args, fl, find_factors_loss)


def fixed_params(model, ps, base_model, args):
    layers = get_layers(base_model)
    stds = np.array([l.kernel.numpy().std() for l in layers])
    ws = np.clip(stds / np.sum(stds) * len(layers) * args.compression_factor, 0, 1)
    weight_dict = {l: w for l, w in zip(layers, ws)}
    fp = FixedParams(
        weight_dict=weight_dict,
        pruning_structure=ps,
        inv_spar_weight_func=lambda x: x / args.sparsity_weight,
        spar_weight_func=lambda x: args.sparsity_weight * x,
    )
    if args.compression_type == "hybrid":
        func = find_factors_params
    elif args.compression_type == "tucker":
        func = find_rank_loss
    else:
        func = find_spar_loss
    ret_dict = compression(model, ps, base_model, args, fp, func)
    if args.compression_type == "tucker":
        ret_dict = {l: (ret_dict[l], 0) for l in ret_dict}
    elif args.compression_type == "spar":
        ret_dict = {l: (0, ret_dict[l]) for l in ret_dict}
    return ret_dict


def main(args):
    ps = get_pruning_structure(args)
    base_model, model = get_resnet(args.input_path)
    model.compile(metrics=["accuracy"])
    if args.method == "fixed_loss":
        pairs = fixed_loss(model, ps, base_model, args)
    else:
        pairs = fixed_params(model, ps, base_model, args)
    ranks = [p[0] for p in pairs]
    spars = [p[1] for p in pairs]
    new_model = get_decomp_resnet(ranks, ps, spars, args.input_path)
    new_model.save(f"{args.output_path}")
    print(f"Compressed model saved to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
