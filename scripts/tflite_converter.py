import os
import tensorflow as tf
import argparse


def convert_to_tflite(new_model, args):
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


def main(args):
    model = tf.keras.models.load_model(args.input_path)
    convert_to_tflite(model, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to TFLite")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input SavedModel"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the compressed TFLite model",
    )
    args = parser.parse_args()
    args.input_path = os.path.abspath(args.input_path)
    args.output_path = os.path.abspath(args.output_path)
    if not os.path.exists(args.input_path):
        parser.error(f"Input path '{args.input_path}' does not exist")
    if not args.input_path[-3:] != ".h5":
        parser.error(
            f"Input path should point to SavedModel in H5 format, got {args.input_path}"
        )
    output_folder = os.path.dirname(args.output_path)
    if not os.path.exists(output_folder):
        parser.error(f"Output folder '{output_folder}' does not exist")
    if args.output_path[-7:] != ".tflite":
        args.output_path = args.output_path + ".tflite"
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
