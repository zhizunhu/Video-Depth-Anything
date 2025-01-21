<div align="center">
<h1>Video Depth Anything</h1>


</div>

This work presents **Video Depth Anything** based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), which can be applied to arbitrarily long videos without compromising quality, consistency, or generalization ability. Compared with other diffusion-based models, it enjoys faster inference speed, fewer parameters, and higher consistent depth accuracy.

![teaser](assets/teaser_video_v2.png)

## News
- **2025-01-25:** Paper, project page, code, models, and demo are all released.


## Pre-trained Models
We provide **two models** of varying scales for robust and consistent video depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 28.4M | Coming soon |
| Depth-Anything-V2-Large | 381.8M | Coming soon |


## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Inference a video
```bash
python3 run.py --input_video ./assets/example_videos/basketball.mp4 --output_dir ./outputs --encoder vitl
```

## Acknowledgement

## LICENSE

## Citation

If you find this project useful, please consider citing:

```bibtex
```


## LICENSE
Video-Depth-Anything-Small model is under the Apache-2.0 license. Video-Depth-Anything-Large model is under the CC-BY-NC-4.0 license.