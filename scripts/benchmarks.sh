#!/bin/bash

# Set the input path and base output path
input_path="../models/resnet_l1.h5"
base_output_path="../models/lite/model_l1"

# Declare an associative array with max_acc_loss values as keys and corresponding output paths as values
declare -A max_acc_loss_paths
max_acc_loss_paths=( ["0.02"]="${base_output_path}_0.02.tflite" ["0.04"]="${base_output_path}_0.04.tflite" ["0.08"]="${base_output_path}_0.08.tflite" )

# Iterate through the max_acc_loss values and run the command with the corresponding output path
for max_acc_loss in "${!max_acc_loss_paths[@]}"; do
    output_path="${max_acc_loss_paths[$max_acc_loss]}"
    python compress.py --input_path="$input_path" --output_path="$output_path" --method=fixed_loss --max_acc_loss="$max_acc_loss"
done
