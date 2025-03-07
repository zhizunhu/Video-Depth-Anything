import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import cv2
import csv
import json
import glob
import shutil
from natsort import natsorted

def gen_json(root_path, start_id, end_id, step, save_path=None):
    data = {}
    data["bonn"] = []
    pieces  = glob.glob(os.path.join(root_path, "*"))
    count = 0
    for piece in pieces:
        if not os.path.isdir(piece):
            continue
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(os.path.join(piece, "rgb/*.png"))
        images = natsorted(images)
        depths = glob.glob(os.path.join(piece, "depth/*.png"))
        depths = natsorted(depths)
        images = images[start_id:end_id:step]
        depths = depths[start_id:end_id:step]
        
        for i in range(len(images)):
            image = images[i]
            xx = image[len(root_path)+1:]
            depth = depths[i][len(root_path)+1:]
            tmp = {}
            tmp["image"] = xx
            tmp["gt_depth"] = depth
            tmp["factor"] = 5000.0
            name_dict[name].append(tmp)
        data["bonn"].append(name_dict)
    with open(save_path, "w") as f:
        json.dump(data, f, indent= 4)  

def even_or_odd(num):
    if num % 2 == 0:
        return num
    else:
        return num - 1

def extract_bonn(
    root,
    depth_root,
    saved_dir,
    sample_len,
    datatset_name,
):
    scenes_names = os.listdir(depth_root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        # load all images
        all_img_names = os.listdir(osp.join(depth_root, seq_name, "rgb"))
        all_img_names = [x for x in all_img_names if x.endswith(".png")]
        print(f"sequence frame number: {len(all_img_names)}")

        all_img_names.sort()
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))

        all_depth_names = os.listdir(osp.join(depth_root, seq_name, "depth"))
        all_depth_names = [x for x in all_depth_names if x.endswith(".png")]
        print(f"sequence depth number: {len(all_depth_names)}")

        # for not zero padding image name
        all_depth_names.sort()
        all_depth_names = sorted(
            all_depth_names, key=lambda x: int(x.split(".")[0][-4:])
        )

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step + 1} / {seq_len//step}")
            video_imgs = []
            video_depths = []
            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(root, seq_name, "rgb", all_img_names[idx])
                depth_path = osp.join(
                    depth_root, seq_name, "depth", all_depth_names[idx]
                )

                img = np.array(Image.open(im_path))
                height, width = img.shape[:2]
                height = even_or_odd(height)
                width = even_or_odd(width)
                img = img[:height, :width, :]
                out_img_path = osp.join(
                    saved_dir, datatset_name,seq_name, "rgb", all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name,seq_name, "depth", all_depth_names[idx]
                )
                os.makedirs(osp.dirname(out_img_path), exist_ok=True)
                os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
                cv2.imwrite(out_img_path, img)
                shutil.copyfile(depth_path, out_depth_path)

    # 110 frames like DepthCraft
    out_json_path = osp.join(saved_dir, datatset_name, "bonn_video.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), 
        start_id=30, end_id=140, step=1, save_path=out_json_path)
    
    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "bonn_video_500.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0, end_id=500, step=1, save_path=out_json_path)


if __name__ == "__main__":
    extract_bonn(
        root="path/to/Bonn-RGBD",
        depth_root="path/to/Bonn-RGBD",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="bonn",
    )