#!/usr/bin/env python

import numpy as np
import re
import sys
import os
import argparse
import cv2
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# load poses from Virtual Kitti 2
def load_poses(path, isLeft):
    f = open(path, "r")
    lines = f.readlines()
    img_name_poses_map = {}
    lines = lines[1:]
    if isLeft:
        beg = 0
    else:
        beg = 1
    for i in range(beg, len(lines), 2):
        line = lines[i].strip()
        tokens = line.split(" ")
        frameId = tokens[0]
        img_name = frameId.zfill(6) #"%06d" % frameId
        r11, r12, r13 = np.float64(tokens[2]), np.float64(tokens[3]), np.float64(tokens[4])
        t1, t2, t3 = np.float64(tokens[5]), np.float64(tokens[9]), np.float64(tokens[13])
        r21, r22, r23 = np.float64(tokens[6]), np.float64(tokens[7]), np.float64(tokens[8])
        r31, r32, r33 = np.float64(tokens[10]), np.float64(tokens[11]), np.float64(tokens[12])

        Rcw = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        tcw = np.array([t1, t2, t3])

        img_name_poses_map[img_name] = (Rcw, tcw)
    f.close()
    return img_name_poses_map

def load_intrinsic(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    line = lines[1]
    line = line.strip()
    tokens = line.split(" ")
    fx, fy, cx, cy = np.float64(tokens[2]), np.float64(tokens[3]), np.float64(tokens[4]), np.float64(tokens[5])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K

def read_pfm(fpath, isCentimeter=False):
    file = open(fpath, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if isCentimeter:
        data /= 100
    return data, shape

def write_pfm(fpath, data, scale=1, file_identifier=b'Pf', dtype="float32"):
    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    # print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)

def compare_depth_maps(start_ID, args, pipeline_depth_map_names, max_depth, flag="raw"):
    if flag == "raw":
        stats_file = os.path.join(args.output_path, "KF_fused_maps/statistics/evaluation_raw_depth_maps.txt")
    elif flag == "kf_fusion":
        stats_file = os.path.join(args.output_path, "KF_fused_maps/statistics/evaluation_kf_fused_depth_maps.txt")
    elif flag == "conf_fusion":
        stats_file = os.path.join(args.output_path, "conf_fused_maps/statistics/evaluation_conf_fused_depth_maps.txt")
    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_valid_pixels = 0

    n_frames = len(pipeline_depth_map_names) - 1
    valid1s = []
    depth_names = []
    for i in tqdm(range(start_ID, n_frames), desc="Loading and comparing raw depth maps", unit="depth maps"):
        depth_name = pipeline_depth_map_names[i][-6:]
        gt_depth_name = "gt_depth_" + depth_name + ".pfm"
        gt_depth_map_path = os.path.join(args.gt_folder, gt_depth_name)
        # depth maps
        gt_depth_map, shape = read_pfm(gt_depth_map_path, isCentimeter=True)
        gt_depth_map[gt_depth_map >= max_depth] = 0

        height, width = shape[0], shape[1]
        depth_names.append(depth_name)
        if flag == "raw":
            raw_depth_name = "depth_" + depth_name + ".pfm"
            raw_depth_map_path = os.path.join(args.src_folder, raw_depth_name)
            pipeline_depth_map, _ = read_pfm(raw_depth_map_path, isCentimeter=True)

            # load kf-based fused depth maps
            kf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            kf_fused_depth_dir = os.path.join(args.output_path, "KF_fused_maps")
            kf_fused_depth_map_path = os.path.join(kf_fused_depth_dir, kf_fused_depth_name)
            kf_fused_depth_map, _ = read_pfm(kf_fused_depth_map_path, isCentimeter=False)

            # load confidence-based fused depth maps
            conf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            conf_fused_depth_dir = os.path.join(args.output_path, "conf_fused_maps")
            conf_fused_depth_map_path = os.path.join(conf_fused_depth_dir, conf_fused_depth_name)
            conf_fused_depth_map, _ = read_pfm(conf_fused_depth_map_path, isCentimeter=False)

            pipeline_depth_map[kf_fused_depth_map == 0] = 0.0
            pipeline_depth_map[conf_fused_depth_map == 0] = 0.0

        elif flag == "kf_fusion":
            # kf depth maps
            conf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            conf_fused_depth_dir = os.path.join(args.output_path, "conf_fused_maps")
            conf_fused_depth_map_path = os.path.join(conf_fused_depth_dir, conf_fused_depth_name)
            conf_fused_depth_map, _ = read_pfm(conf_fused_depth_map_path, isCentimeter=False)

            kf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            kf_fused_depth_dir = os.path.join(args.output_path, "KF_fused_maps")
            kf_fused_depth_map_path = os.path.join(kf_fused_depth_dir, kf_fused_depth_name)
            pipeline_depth_map, _ = read_pfm(kf_fused_depth_map_path, isCentimeter=False)

            pipeline_depth_map[conf_fused_depth_map == 0.0] = 0.0

        else:
            # conf depth maps
            conf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            conf_fused_depth_dir = os.path.join(args.output_path, "conf_fused_maps")
            conf_fused_depth_map_path = os.path.join(conf_fused_depth_dir, conf_fused_depth_name)
            pipeline_depth_map, _ = read_pfm(conf_fused_depth_map_path, isCentimeter=False)

            kf_fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            kf_fused_depth_dir = os.path.join(args.output_path, "KF_fused_maps")
            kf_fused_depth_map_path = os.path.join(kf_fused_depth_dir, kf_fused_depth_name)
            kf_fused_depth_map, _ = read_pfm(kf_fused_depth_map_path, isCentimeter=False)

            pipeline_depth_map[kf_fused_depth_map == 0.0] = 0.0

        both_valid = (pipeline_depth_map != 0) & (gt_depth_map != 0)
        valid1 = np.sum(pipeline_depth_map != 0)
        valid1s.append(valid1)

        valid2 = np.sum(gt_depth_map != 0)
        # print(np.sum(pipeline_depth_map != 0))
        both_zeros = (pipeline_depth_map == 0) & (gt_depth_map == 0)

        if np.sum(both_valid) == 0:
            continue
        median = np.median(np.absolute(pipeline_depth_map[both_valid] - gt_depth_map[both_valid]))
        medians.append(median)
        mae = np.mean(np.absolute(pipeline_depth_map[both_valid] - gt_depth_map[both_valid]))
        maes.append(mae)

        global_min_mae = min(mae, global_min_mae)
        global_max_mae = max(mae, global_max_mae)

        N_both_valid = np.sum(both_valid)
        num_pixels.append(N_both_valid)
        total_both_valid_pixels += N_both_valid

    weighted_average_mae = 0.0
    with open(stats_file, 'w') as f:
        for i in range(len(maes)):
            f.write("No.{0}: {1} \n".format(i+start_ID, depth_names[i]))
            f.write("there are {0} pixels have depth\n".format(valid1s[i]))
            f.write("MAE: {0} over {1} pixels \n".format(maes[i], num_pixels[i]))
            f.write("Median of error: {0} over {1} pixels \n".format(medians[i], num_pixels[i]))
            weighted_average_mae += (maes[i] * (num_pixels[i] / total_both_valid_pixels))
            # if flag is not "raw":
            #     f.write("There are {0} points are fused and {1} points are new\n".format(points_fused_counters[i], points_new_counters[i]))
            f.write(
                "=====================================================================================================\n")
        f.write("Minimum MAE: {0}, Median Medians: {1}, weighted average of MAE {2}, Maximum MAE: {3}\n".format(
            global_min_mae, np.median(medians), weighted_average_mae, global_max_mae))
    f.close()

def write_vis_png(fpath, data, max_value):
    # vis_img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # # groundtruth_color = cv2.cvtColor(groundtruth, cv2.COLOR_GRAY2RGB)
    # # Saving the image
    # cv2.imwrite(fpath, vis_img)
    fig, ax1 = plt.subplots(1, 1)
    norm = mcolors.Normalize(vmin=0, vmax=max_value)
    # For plotting COLMAP
    im1 = ax1.imshow(data, norm=norm, cmap=cm.jet)
    ax1.set_title("Trace of Uncertainty")
    plt.colorbar(im1, ax=ax1)
    plt.savefig(fpath)
    plt.close()


