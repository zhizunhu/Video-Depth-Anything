#!/bin/sh
set -x
set -e

pred_disp_root=/path/to/saved/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
benchmark_root=/path/to/gt_depth/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

#eval sintel

python benchmark/eval/eval_tae.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets sintel \
    --start_idx 0 \
    --eval_scenes_num 23

#eval scannet
python benchmark/eval/eval_tae.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets scannet \
    --start_idx 10 \
    --end_idx 180 \
    --eval_scenes_num 20 \
    --hard_crop


