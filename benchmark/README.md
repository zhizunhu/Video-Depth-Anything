# BENCHMARK

## Prepare Dataset
Download datasets from the following links:
[sintel](http://sintel.is.tue.mpg.de/) [kitti](https://www.cvlibs.net/datasets/kitti/) [bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) [scannet](http://www.scan-net.org/) [nyuv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

```bash
dataset=sintel/kitti/bonn/scannet/nyuv2
cd benchmark/dataset_extract
python dataset_extrtact${dataset}.py
```

## Run inference
```bash
bash benchmark/infer/infer.sh
```

## Run evaluation
```bash
## tae
bash benchmark/eval/eval_tae.sh
## ~110frame like DepthCrafter
bash benchmark/eval/eval.sh
## ~500frame 
bash benchmark/eval/eval_500.sh
```