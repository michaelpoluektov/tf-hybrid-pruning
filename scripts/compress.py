import sys
import os

sys.path.append("../src")

import argparse
from pruning import find_factors_loss, find_factors_params
from model import get_resnet, get_decomp_resnet
from utils import FixedLoss, FixedParams, Eval, PruningStructure
import numpy as np
import tensorflow as tf


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
        def reduce_ker(k): return np.mean(k, axis = 2)
        def transform_mask
    pass


def fixed_loss(model, base_model, args):

    pass


def fixed_params(model, base_model, args):
    pass


def main(args):
    base_model, model = get_resnet(args.input_path)
    if args.method == "fixed_loss":
        pairs = fixed_loss(model, base_model, args)
    else:
        pairs = fixed_params(model, base_model, args)
    ranks = [p[0] for p in pairs]
    spars = [p[1] for p in pairs]
    new_model = get_decomp_resnet(ranks, spars, args.input_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    tflite_model = converter.convert()

    # Save the TFLite model to output_path
    with open(args.output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Compressed model saved to {args.output_path}")
