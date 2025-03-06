#!/bin/sh
set -x
set -e

pred_disp_root=/path/to/saved/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
benchmark_root=/path/to/gt_depth/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

#eval scannet
python benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets scannet_500

#eval kitti
python benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets kitti_500

#eval bonn
python benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets bonn_500

#eval nyu
python benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets nyuv2_500
