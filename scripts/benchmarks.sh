#!/bin/bash

# Set the base input path and base output path
base_input_path="../models"
base_output_path="../models/keras"

# Declare an array with model names
models=("resnet_l1" "resnet_bn_finetune")

# Declare an array with extra flags
extra_flags=(
  "--pruning_structure=block --block_size=8,8"
  "--pruning_structure=unstructured"
  "--pruning_structure=filter"
  "--pruning_structure=channel"
)

# Iterate through the models
for model in "${models[@]}"; do
  input_path="${base_input_path}/${model}.h5"

  # Iterate through the extra flags
  for flags in "${extra_flags[@]}"; do
    flag_suffix=$(echo "$flags" | sed 's/[^a-zA-Z0-9_-]/_/g')

    # Iterate through the max_acc_loss values from 0.02 to 0.10, with an increment of 0.02
    for max_acc_loss in $(seq 0.02 0.02 0.10); do
      output_path="${base_output_path}/${model}_${flag_suffix}_$(printf '%.2f' "$max_acc_loss").tflite"
      python compress.py --input_path="$input_path" --output_path="$output_path" --method=fixed_loss --max_acc_loss="$max_acc_loss" $flags
    done
  done
done
