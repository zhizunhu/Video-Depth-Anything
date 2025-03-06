#!/bin/sh
set -x
set -e


out_path=/path/to/saved/root_directory
json_file=/path/to/dataset_json
dataset=which_dataset

#infer
python benchmark/infer/infer.py \
    --infer_path $out_path \
    --json_file $json_file \
    --datasets $dataset
