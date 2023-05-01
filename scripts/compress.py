import sys
import os

sys.path.append("../src")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pruning import find_factors_loss, find_factors_params
from model import get_resnet, get_decomp_resnet
from dataset import get_dataset
from utils import (
    FixedLoss,
    FixedParams,
    Eval,
    PruningStructure,
    get_decomp_weight,
    get_weight,
    test_weights_eval,
)
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimise model and convert to TFLite")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input model weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the compressed TFLite model",
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
        type=tuple,
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


def fixed_loss(model, ps, base_model, args):
    fl = FixedLoss(
        ps,
        eval_func=get_loss_eval_func(args),
        inv_spar_weight_func=lambda x: x / args.sparsity_weight,
        spar_weight_func=lambda x: args.sparsity_weight * x,
    )
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
    comp_dict = find_factors_loss(base_model, layers, ev, fl)
    _, new_val = model.evaluate(val_ds, verbose=0)
    print(
        f"Found factors. Accuracy loss: test={(test_accuracy - ev.base_accuracy)*100:.2}%, val={(val_accuracy-new_val)*100:.2f}%"
    )
    print("FACTORS:")
    for l in comp_dict:
        r, s = comp_dict[l]
        print(
            f"{l.name}: rank={r} ({fl.decomp_weight_func(get_decomp_weight(l, r) / get_weight(l))*100:.2f}%), sparsity={s:.2f}% ({fl.spar_weight_func(s):.2f}%)"
        )
    pairs = [comp_dict[l] for l in layers]
    return pairs


def fixed_params(model, ps, base_model, args):
    layers = get_layers(base_model)
    test_ds, val_ds = get_dataset(False, 224, 1, 16, True)
    print("Evaluating model...", end=" ")
    _, val_accuracy = model.evaluate(val_ds, verbose=0)
    _, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy: val={val_accuracy*100:.2f}%, test={test_accuracy*100:.2f}%")
    stds = np.array([l.kernel.numpy().std() for l in layers])
    ws = np.clip(stds / np.sum(stds) * len(layers) * args.compression_factor, 0, 1)
    weight_dict = {l: w for l, w in zip(layers, ws)}
    fp = FixedParams(
        weight_dict=weight_dict,
        pruning_structure=ps,
        inv_spar_weight_func=lambda x: x / args.sparsity_weight,
        spar_weight_func=lambda x: args.sparsity_weight * x,
    )
    ev = Eval(
        model=model,
        ds=test_ds,
        pbar=tqdm(total=len(layers)),
        base_accuracy=test_accuracy,
    )
    comp_dict = find_factors_params(base_model, layers, ev, fp)
    _, new_val = model.evaluate(val_ds, verbose=0)
    print(
        f"Found factors. Accuracy loss: test={(test_accuracy - ev.base_accuracy)*100:.2}%, val={(val_accuracy-new_val)*100:.2f}%"
    )
    print("FACTORS:")
    for l in comp_dict:
        r, s = comp_dict[l]
        print(
            f"{l.name}: rank={r} ({fp.decomp_weight_func(get_decomp_weight(l, r) / get_weight(l))*100:.2f}%), sparsity={s:.2f}% ({fp.spar_weight_func(s):.2f}%)"
        )
    pairs = [comp_dict[l] for l in layers]
    return pairs


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
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    # Save the TFLite model to output_path
    with open(args.output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Compressed model saved to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
