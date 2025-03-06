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


# def _read_image(img_rel_path) -> np.ndarray:
#     image_to_read = img_rel_path
#     image = Image.open(image_to_read)  # [H, W, rgb]
#     image = np.asarray(image)
#     return image


# def depth_read(filename):
#     depth_in = _read_image(filename)
#     #depth_decoded = depth_in / 1000.0
#     return depth_in

def gen_json(root_path, start_id, end_id, step, save_path=None, original=False):
    data = {}
    data["scannet"] = []
    pieces  = glob.glob(os.path.join(root_path, "*"))

    color = 'color' if not original else 'color_origin'

    for piece in pieces:
        if not os.path.isdir(piece):
            continue
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(os.path.join(piece,color, "*.jpg"))
        images = natsorted(images)
        depths = glob.glob(os.path.join(piece, "depth/*.png"))
        depths = natsorted(depths)
        images = images[start_id:end_id:step]
        depths = depths[start_id:end_id:step]
        print(f"sequence frame number: {piece}")
        # images = images[30:140]
        # depths = depths[30:140]
        count = 0
        for i in range(len(images)):
            image = images[i]
            xx = image[len(root_path)+1:]
            depth = depths[i][len(root_path)+1:]
            
            base_path = os.path.dirname(image)
            base_path = base_path.replace(color, 'intrinsic')
            K = np.loadtxt(base_path + '/intrinsic_depth.txt')

            pose_path = image.replace(color, 'pose').replace('.jpg', '.txt')
            pose = np.loadtxt(pose_path)
            
            tmp = {}
            tmp["image"] = xx
            tmp["gt_depth"] = depth
            tmp["factor"] = 1000.0
            tmp["K"] = K.tolist()
            tmp["pose"] = pose.tolist()
            name_dict[name].append(tmp)

            # count += 1
            # if count > 300:
            #     break
        data["scannet"].append(name_dict)
        
    with open(save_path, "w") as f:
        json.dump(data, f, indent= 4)    

def extract_scannet(
    root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    # scenes_names = os.listdir(root)
    # scenes_names = sorted(scenes_names)[:100]
    # all_samples = []
    # for i, seq_name in enumerate(tqdm(scenes_names)):
    #     all_img_names = os.listdir(osp.join(root, seq_name, "color"))
    #     all_img_names = [x for x in all_img_names if x.endswith(".jpg")]
    #     all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0]))
    #     all_img_names = all_img_names[:510]
    #     print(f"sequence frame number: {len(all_img_names)}")

    #     seq_len = len(all_img_names)
    #     step = sample_len if sample_len > 0 else seq_len

    #     for ref_idx in range(0, seq_len, step):
    #         print(f"Progress: {seq_name}, {ref_idx // step + 1} / {seq_len//step}")

    #         video_imgs = []
    #         video_depths = []

    #         if (ref_idx + step) <= seq_len:
    #             ref_e = ref_idx + step
    #         else:
    #             continue

    #         for idx in range(ref_idx, ref_e):
    #             im_path = osp.join(root, seq_name, "color", all_img_names[idx])
    #             depth_path = osp.join(
    #                 root, seq_name, "depth", all_img_names[idx][:-3] + "png"
    #             )
    #             pose_path = osp.join(
    #                 root, seq_name, "pose", all_img_names[idx][:-3] + "txt"
    #             )

    #             #depth = depth_read(depth_path)
    #             img = np.array(Image.open(im_path))
    #             origin_img = img.copy()

    #             img = img[8:-8, 11:-11, :]
    #             #depth = depth[8:-8, 11:-11]
    #             out_img_path = osp.join(
    #                 saved_dir, datatset_name, seq_name, "color", all_img_names[idx]
    #             )
    #             out_img_origin_path = osp.join(
    #                 saved_dir, datatset_name, seq_name, "color_origin", all_img_names[idx]
    #             )
    #             out_depth_path = osp.join(
    #                 saved_dir, datatset_name, seq_name, "depth", all_img_names[idx][:-3] + "png"
    #             )
    #             out_pose_path = osp.join(
    #                 saved_dir, datatset_name, seq_name, "pose", all_img_names[idx][:-3] + "txt"
    #             )
    #             os.makedirs(osp.dirname(out_img_path), exist_ok=True)
    #             os.makedirs(osp.dirname(out_img_origin_path), exist_ok=True)
    #             os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
    #             os.makedirs(osp.dirname(out_pose_path), exist_ok=True)
    #             cv2.imwrite(
    #                 out_img_path,
    #                 img,
    #             )
    #             cv2.imwrite(
    #                 out_img_origin_path,
    #                 origin_img,
    #             )
    #             shutil.copyfile(depth_path, out_depth_path)
    #             shutil.copyfile(pose_path, out_pose_path)
            
    #         intrinsic_path = osp.join(
    #             root, seq_name, "intrinsic", "intrinsic_depth.txt"
    #         )
    #         out_intrinsic_path = osp.join(
    #             saved_dir, datatset_name, seq_name, "intrinsic", "intrinsic_depth.txt"
    #         )
    #         os.makedirs(osp.dirname(out_intrinsic_path), exist_ok=True)
    #         shutil.copyfile(intrinsic_path, out_intrinsic_path)

    # 90 frames like DepthCraft
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0,
        end_id=90*3,
        step=3,
        save_path=out_json_path,
    )      

    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video_500.json")    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0,
        end_id=500,
        step=1,
        save_path=out_json_path,
    )

    # tae 
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video_tae.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0,
        end_id=192,
        step=1,
        save_path=out_json_path,
        original=True,
    )

                




if __name__ == "__main__":
    extract_scannet(
        root="/mnt/bn/mobile-depth-data/video_dataset/scannet_data/scannet1",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="scannet",
    )