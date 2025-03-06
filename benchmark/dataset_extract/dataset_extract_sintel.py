# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# # Data loading based on https://github.com/NVIDIA/flownet2-pytorch


import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio
import cv2
import json
import glob
import shutil


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N


def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth

def gen_json(root_path, out_path):
    data = {}
    data["sintel"] = []

    pieces  = glob.glob(os.path.join(root_path, "clean/*"))

    for piece in pieces:
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(os.path.join(piece, "*.png"))

        for image in images:
            re_path = image[len(root_path)+1:]
            
            depth = re_path.replace("clean", "depth")
            mask = re_path.replace("clean", "rigidity")
            cam_pose = image.replace("clean", "camdata_left").replace('.png', '.cam')

            M1, N1 = cam_read(cam_pose)

            T_1 = np.eye(4)
            T_1[:3,:] = N1

            T_1 = np.linalg.inv(T_1)
            
            tmp = {}
            tmp["image"] = re_path
            tmp["gt_depth"] = depth
            tmp['mask'] = mask
            tmp["K"] = M1.tolist()
            tmp["pose"] = T_1.tolist()
            tmp["factor"] = 65535 / 650

            name_dict[name].append(tmp)
        data["sintel"].append(name_dict)
        
    with open(os.path.join(root_path, "sintel_video.json"), "w") as f:
        json.dump(data, f, indent= 4)  

def extract_sintel(
    root,
    depth_root,
    pose_root,
    rigidity_root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    scenes_names = os.listdir(root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = os.listdir(os.path.join(root, seq_name))
        all_img_names = [x for x in all_img_names if x.endswith(".png")]
        all_img_names.sort()
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step} / {seq_len // step}")

            video_imgs = []
            video_depths = []

            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(root, seq_name, all_img_names[idx])
                depth_path = osp.join(
                    depth_root, seq_name, all_img_names[idx][:-3] + "dpt"
                )
                cam_path = osp.join(
                    pose_root, seq_name, all_img_names[idx][:-3] + "cam"
                )
                rigidity_path = osp.join(
                    rigidity_root, seq_name, all_img_names[idx][:-3] + "png"
                )
                 
                # depth = depth_read(depth_path)
                # depth = depth * 65535 / 650
                depth_path = depth_path = osp.join(
                    depth_root, seq_name, all_img_names[idx][:-3] + "png"
                )
                depth = np.asarray(Image.open(depth_path))
                img = np.array(Image.open(im_path))
                
                out_img_path = osp.join(
                    saved_dir, datatset_name,'clean', seq_name, all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name,'depth', seq_name, all_img_names[idx][:-3] + "png"
                )
                out_rigidity_path = osp.join(
                    saved_dir, datatset_name,'rigidity', seq_name, all_img_names[idx]
                )
                out_cam_path = osp.join(
                    saved_dir, datatset_name,'camdata_left', seq_name, all_img_names[idx][:-3] + "cam"
                )
                os.makedirs(osp.dirname(out_img_path), exist_ok=True)
                os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
                os.makedirs(osp.dirname(out_cam_path), exist_ok=True)
                os.makedirs(osp.dirname(out_rigidity_path), exist_ok=True)

                cv2.imwrite(
                    out_img_path,
                    img,
                )
                cv2.imwrite(
                    out_depth_path,
                    depth.astype(np.uint16)
                )
                shutil.copyfile(cam_path, out_cam_path)
                if os.path.exists(rigidity_path):
                    shutil.copyfile(rigidity_path, out_rigidity_path)
    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name),
        out_path=osp.join(saved_dir, datatset_name, "sintel_video.json"))
        

                

            

            


if __name__ == "__main__":
    extract_sintel(
        root="/mnt/bn/zfhhuman/sintel/raw/training/clean",
        depth_root="/mnt/bn/zfhhuman/sintel/depth",
        pose_root='/mnt/bn/omnidata/video_benchmark_tae/sintel/camdata_left/',
        rigidity_root='/mnt/bn/omnidata/video_benchmark_tae/sintel/rigidity/',
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="sintel",
    )