#!/bin/bash

# Set the base input path and base output path
base_input_path="../models"
base_output_path="../models/keras"

# Declare an array with model names
models=("resnet_l1" "resnet_bn_finetune")

# Declare an array with extra flags and their corresponding abbreviations
declare -A extra_flags
declare -A ds_sizes

extra_flags=(
  ["block_8x8"]="--pruning_structure=block --block_size=8,8"
  ["unstruct"]="--pruning_structure=unstructured"
  ["filter"]="--pruning_structure=filter"
  ["channel"]="--pruning_structure=channel"
  # ["tucker"]="--compression_type=tucker"
)

ds_sizes=(
  ["ds_10"]="--dataset_size=10"
  ["ds_20"]="--dataset_size=20"
  ["ds_40"]="--dataset_size=40"
  ["ds_80"]="--dataset_size=80"
  ["ds_160"]="--dataset_size=160"
)

# Iterate through the models
for model in "${models[@]}"; do
  input_path="${base_input_path}/${model}.h5"

  # Iterate through the extra flags
  # for flag_suffix in "${!extra_flags[@]}"; do
  for flag_suffix in "${!ds_sizes[@]}"; do
    # flags="${extra_flags[$flag_suffix]}"
    flags="${ds_sizes[$flag_suffix]}"

    echo $flags
    echo $flag_suffix
    # Iterate through the max_acc_loss values from 0.02 to 0.10, with an increment of 0.02
    for max_acc_loss in $(seq 0.02 0.02 0.02); do
      output_path="${base_output_path}/${model}_${flag_suffix}_$(printf '%.2f' "$max_acc_loss").h5"
      
      # Check if the output file already exists
      if [ -e "$output_path" ]; then
        echo "Output file $output_path already exists. Skipping..."
      else
        echo "Working on file $output_path..."
        python compress.py --input_path="$input_path" --output_path="$output_path" --method=fixed_loss --sparsity_weight=4 --max_acc_loss="$max_acc_loss" $flags
      fi
    done
  done
done
