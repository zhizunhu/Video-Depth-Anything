import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import cv2
import json
import glob
from natsort import natsorted
import shutil


def gen_json(root_path, start_id, end_id, step, save_path=None):
    data = {}
    data["nyuv2"] = []
    
    pieces  = glob.glob(os.path.join(root_path, "*"))

    for piece in pieces:
        if not os.path.isdir(piece):
            continue
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(os.path.join(piece, "rgb/*.jpg"))
        images = natsorted(images)
        depths = glob.glob(os.path.join(piece, "depth/*.png"))
        depths = natsorted(depths)
        images = images[start_id:end_id:step]
        depths = depths[start_id:end_id:step]
        # images = images[30:140]
        # depths = depths[30:140]
        count = 0
        for i in range(len(images)):
            image = images[i]
            xx = image[len(root_path)+1:]
            depth = depths[i][len(root_path)+1:]
            
            tmp = {}
            tmp["image"] = xx
            tmp["gt_depth"] = depth
            tmp["factor"] = 6000.0
            name_dict[name].append(tmp)

            # count += 1
            # if count > 300:
            #     break
        data["nyuv2"].append(name_dict)
        
    with open(save_path, "w") as f:
        json.dump(data, f, indent= 4)    

def extract_nyuv2(
    root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    scenes_names = os.listdir(root)
    scenes_names = sorted(scenes_names)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = os.listdir(osp.join(root, seq_name, "rgb"))
        all_img_names = [x for x in all_img_names if x.endswith(".jpg")]
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0]))
        #all_img_names = all_img_names[:scene_frames_len:stride]
        print(f"sequence frame number: {len(all_img_names)}")

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
                    root, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )
            
                depth = Image.open(im_path)
                img = np.array(Image.open(im_path))

                img = img[45:471, 41:601, :]
                #depth = depth[45:471, 41:601]
                out_img_path = osp.join(
                    saved_dir, datatset_name, seq_name, "rgb", all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name, seq_name, "depth", all_img_names[idx]
                )
         
                os.makedirs(osp.dirname(out_img_path), exist_ok=True)
                os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
                
                cv2.imwrite(
                    out_img_path,
                    img,
                )
                cv2.imwrite(
                    out_depth_path,
                    depth.astype(np.uint16),
                )

    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "nyuv2_video_500.json")    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0,
        end_id=500,
        step=1,
        save_path=out_json_path,
    )

                




if __name__ == "__main__":
    # we use matlab to extract 8 scenes from NYUv2
    #--basement_0001a, bookstore_0001a, cafe_0001a, classroom_0001a, kitchen_0003, office_0004, playroom_0002, study_0002
    extract_scannet(
        root="path/to/nyuv2",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="nyuv2",
    )