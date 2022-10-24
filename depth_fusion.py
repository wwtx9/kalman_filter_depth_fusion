#!/usr/bin/env python
import argparse
from plistlib import FMT_XML
import numpy as np
import os
import re
import pylab as plt
from tqdm import tqdm
import open3d as o3d
import quaternion
from numpy.linalg import inv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--src_folder",
                        help="path to pipeline's depth maps", type=str, default="")
    parser.add_argument("-r2", "--src_poses_path",
                        help="path to pipeline's poses", type=str, default="")
    parser.add_argument("-n", "--dataset_name",
                        help="dataset name", type=str, default="FLORIDA")
    parser.add_argument("-t", "--gt_folder",
                        help="path to ground truth depth map", type=str)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                        help="Output path where all metrics and results will be stored.")
    args = parser.parse_args()
    return args

def load_poses(path):
    f = open(path, "r")
    lines = f.readlines()
    img_name_poses_map = {}

    for line in lines:
        tokens = line.split(" ")
        timestamp = tokens[0]
        timestamp_tokens = timestamp.split(".")
        img_name = timestamp_tokens[0]+timestamp_tokens[1]
        twc = np.array([np.float64(tokens[1]), np.float64(tokens[2]), np.float64(tokens[3])])
        qwc = np.quaternion(np.float64(tokens[7]), np.float64(tokens[4]), np.float64(tokens[5]), np.float64(tokens[6]))
        Rwc = quaternion.as_rotation_matrix(qwc)
        Rcw = np.transpose(Rwc)
        tcw = -Rcw.dot(twc)
        img_name_poses_map[img_name] = (Rcw, tcw)
    f.close()
    return img_name_poses_map

def proj(global_map, pose, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    for i in range(len(global_map)):
        Rcw, tcw = pose[0], pose[1]
        landmark = global_map[i]
        Xc = Rcw.dot(landmark) + tcw

        u_proj = fx * Xc[0] / Xc[2] + cx
        v_proj = fy * Xc[1] / Xc[2] + cy
        yield i, round(u_proj), round(v_proj)

def iproj(pose, K, u, v, depth):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    Xc = np.array([x, y, z])

    Rcw, tcw = pose[0], pose[1]
    Rwc = np.transpose(Rcw)
    twc = -Rwc.dot(tcw)

    Xw = Rwc.dot(Xc) + twc
    return Xw

def makeJacobian(u, v, Z, K):
    #P(u, v, Z) = P(ui, vi, Zi) + Ji(u-ui, v-vi, Z-Zi)
    #P = [X, Y, Z]
    # X = Z*(u - cx)/fx
    # Y = Z*(v - cy)/fy
    #Ji = [dX/du, dX/dv, dX/dZ,
    #      dY/du, dY/dv, dY/dZ,
    #      dZ/du, dZ/dv, dZ/dZ]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.array([[Z/fx, 0, (u-cx)/fx],
                  [0, Z/fy, (v-cy)/fy],
                  [0, 0, 1]])
    return J

def makeQ(depth):
    mvInvLevelSigma2 = np.ones(8)
    mvLevelSigma2 = np.ones(8)
    mvScaleFactor = np.ones(8)
    mvInvScaleFactor = np.ones(8)

    max_depth = 10

    for i in range(1, 8):
        mvScaleFactor[i] = mvScaleFactor[i - 1] * 1.2
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i]

    for i in range(8):
        mvInvScaleFactor[i] = 1.0 / mvScaleFactor[i]
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i]

    level = round(depth * 7 / max_depth)


    Q = mvLevelSigma2[level] * np.identity(3)

    return Q

def computeU(u, v ,depth, K, pose):
    J = makeJacobian(u, v, depth, K)
    Q = makeQ(depth)
    Rcw = pose[0]
    Rwc = np.transpose(Rcw)

    U_obs = np.linalg.multi_dot([Rwc, J, Q, J.T, Rwc.T])

    return U_obs

def read_pipeline_depth_maps(path):
    file = open(path, 'rb')

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
    return data, shape

def read_colmap_depth_maps(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def kf_depth_fusion(X_prior, X_obs, U_prior, U_obs):
    # U_post = (inv(U_obs) + inv(X_obs))
    # U_post = inv(U_post)
    Si = U_prior + U_obs
    Wi = U_prior.dot(inv(Si)) # kalman gain
    ei = X_obs - X_prior
    #update
    X_post = X_prior + Wi.dot(ei)
    U_post = U_prior - Wi.dot(U_prior)
    return X_post, U_post

def compare_raw_depth_maps(tgt_folder, pipeline_depth_map_paths, pipeline_depth_map_names):
    stats_file = os.path.join(args.output_path, "evaluation_raw_depth_maps.txt")
    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_no_zeros_pixels = 0

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and comparing raw depth maps", unit="depth maps"):
        colmap_depth_map_path = tgt_folder + pipeline_depth_map_names[i] + '.png.geometric.bin'
        # depth maps
        colmap_depth_map = read_colmap_depth_maps(colmap_depth_map_path)

        pipline_depth_map_path = pipeline_depth_map_paths[i]
        pipeline_depth_map, _ = read_pipeline_depth_maps(pipline_depth_map_path)

        both_non_zeros = (pipeline_depth_map != 0) & (colmap_depth_map != 0)
        both_zeros = (pipeline_depth_map == 0) & (colmap_depth_map == 0)
        if np.sum(both_non_zeros) == 0:
            continue
        median = np.median(np.absolute(pipeline_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        medians.append(median)
        mae = np.mean(np.absolute(pipeline_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        maes.append(mae)

        global_min_mae = min(mae, global_min_mae)
        global_max_mae = max(mae, global_max_mae)

        N_both_non_zeros = np.sum(both_non_zeros)
        num_pixels.append(N_both_non_zeros)
        total_both_no_zeros_pixels += N_both_non_zeros

    weighted_average_mae = 0.0
    with open(stats_file, 'w') as f:
        for i in range(len(maes)):
            f.write("No.{0} \n".format(i))
            f.write("MAE: {0} over {1} pixels \n".format(maes[i], num_pixels[i]))
            f.write("Median of error: {0} over {1} pixels \n".format(medians[i], num_pixels[i]))
            weighted_average_mae += (maes[i] * (num_pixels[i] / total_both_no_zeros_pixels))
            f.write(
                "=====================================================================================================\n")
        f.write("Minimum MAE: {0}, Median Medians: {1}, weighted average of MAE {2}, Maximum MAE: {3}\n".format(
            global_min_mae, np.median(medians), weighted_average_mae, global_max_mae))
    f.close()

def compare_fused_depth_maps(tgt_folder, img_SE3_map, K, pipeline_depth_map_names, global_map):
    stats_file = os.path.join(args.output_path, "evaluation_fused_depth_maps.txt")
    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_no_zeros_pixels = 0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and comparing fused depth maps", unit="depth maps"):
        colmap_depth_map_path = tgt_folder + pipeline_depth_map_names[i] +'.png.geometric.bin'
        # depth maps
        colmap_depth_map = read_colmap_depth_maps(colmap_depth_map_path)

        pipeline_fused_depth_map = np.zeros((600, 800))
        # pipeline's 3D model
        for point in global_map:
            Rcw = img_SE3_map[pipeline_depth_map_names[i]][0]
            tcw = img_SE3_map[pipeline_depth_map_names[i]][1]
            Xc = Rcw.dot(point) + tcw
            if Xc[2] <= 0:
                continue
            u_proj = fx * Xc[0] / Xc[2] + cx
            v_proj = fy * Xc[1] / Xc[2] + cy
            u_proj, v_proj = round(u_proj), round(v_proj)

            if u_proj < 0 or u_proj >= 800 or v_proj < 0 or v_proj >= 600:
                continue
            pipeline_fused_depth_map[v_proj, u_proj] = Xc[2]

        both_non_zeros = (pipeline_fused_depth_map != 0) & (colmap_depth_map != 0)
        both_zeros = (pipeline_fused_depth_map == 0) & (colmap_depth_map == 0)
        if np.sum(both_non_zeros) == 0:
            continue
        median = np.median(np.absolute(pipeline_fused_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        medians.append(median)
        mae = np.mean(np.absolute(pipeline_fused_depth_map[both_non_zeros] - colmap_depth_map[both_non_zeros]))
        maes.append(mae)

        global_min_mae = min(mae, global_min_mae)
        global_max_mae = max(mae, global_max_mae)

        N_both_non_zeros = np.sum(both_non_zeros)
        num_pixels.append(N_both_non_zeros)
        total_both_no_zeros_pixels += N_both_non_zeros

    weighted_average_mae = 0.0
    with open(stats_file, 'w') as f:
        for i in range(len(maes)):
            f.write("No.{0} \n".format(i))
            f.write("MAE: {0} over {1} pixels \n".format(maes[i], num_pixels[i]))
            f.write("Median of error: {0} over {1} pixels \n".format(medians[i], num_pixels[i]))
            weighted_average_mae += (maes[i]*(num_pixels[i]/total_both_no_zeros_pixels))
            f.write(
                "=====================================================================================================\n")
        f.write("Minimum MAE: {0}, Median Medians: {1}, weighted average of MAE {2}, Maximum MAE: {3}\n".format(global_min_mae, np.median(medians), weighted_average_mae,global_max_mae))
    f.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()

    dataset = args.dataset_name
  
    # Read depth/normal maps from folder
    if dataset == "FLORIDA":
        K = np.array([[595.58148, 0, 380.47882], [0, 593.81262, 302.91428], [0, 0, 1]])
    elif dataset == "MEXICO":
        K = np.array([[600.8175, 0, 383.27002], [0, 600.82904, 287.87112], [0, 0, 1]])
    elif dataset == 'STAVRONIKITA':
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.3586, 313.35413], [0, 0, 1]])
    elif dataset == "REEF":
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.9395, 313.69763], [0, 0, 1]])
    elif dataset == "SPAMIR":
        K = np.array([[1289.2439, 0, 279.92609], [0, 1290.9395, 313.69763], [0, 0, 1]])

    img_name_SE3 = load_poses(args.src_poses_path)
    assert (len(img_name_SE3) == len(pipeline_depth_map_names))

    global_map = []
    global_uncertainties = []

    pipeline_depth_map_names = pipeline_depth_map_names[0:5]

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Start to depth fusion & a build map", unit="depth maps"):
        pipline_depth_map_path = pipeline_depth_map_paths[i]
        pipeline_depth_map, shape = read_pipeline_depth_maps(pipline_depth_map_path)
        pose = img_name_SE3[pipeline_depth_map_names[i]]

        height, width = shape[0], shape[1]
        if len(global_map) == 0:
            for v in range(0, height):
                for u in range(0, width):
                    depth = pipeline_depth_map[v, u]
                    if depth <= 0:
                        continue
                    X_obs = iproj(pose, K, u, v, depth)
                    global_map.append(X_obs)
                    U_obs = computeU(u, v, depth, K, pose)
                    global_uncertainties.append(U_obs)
        else:
            t_map = []
            t_uncertainties = []
            visited = np.zeros((height, width), dtype=bool)
            for (id, u_proj, v_proj) in proj(global_map, pose, K):
                if u_proj < 0 or u_proj >= width or v_proj < 0 or v_proj >= height:
                    continue
                depth = pipeline_depth_map[v_proj, u_proj]
                if depth <= 0 or visited[v_proj, u_proj]:
                    continue
                visited[v_proj, u_proj] = True

                X_obs = iproj(pose, K, u_proj, v_proj, depth)
                U_obs = computeU(u_proj, v_proj, depth, K, pose)
                U_prior = global_uncertainties[id]
                X_prior = global_map[id]
                X_post, U_post = kf_depth_fusion(X_prior, X_obs, U_prior, U_obs)
                # update 3D position and uncertainty
                global_map[id] = X_post
                global_uncertainties[id] = U_post
            # add rest of pixel's 3D position into map
            for v in range(0, height):
                for u in range(0, width):
                    if visited[v, u]:
                        continue
                    X_obs = iproj(pose, K, u_proj, v_proj, depth)
                    U_obs = computeU(u_proj, v_proj, depth, K, pose)
                    t_map.append(X_obs)
                    t_uncertainties.append(U_obs)

            global_map.extend(t_map)
            global_uncertainties.extend(t_uncertainties)
        # print("Complete depth map {0}:{1}".format(i, pipeline_depth_map_names[i]))

    # Evaluate depth maps
    compare_raw_depth_maps(args.gt_folder, pipeline_depth_map_paths, pipeline_depth_map_names)
    compare_fused_depth_maps(args.gt_folder, img_name_SE3, K, pipeline_depth_map_names, global_map)

    # Pass global_map to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # landmarks = np.asarray(global_map)
    # pcd.points = o3d.utility.Vector3dVector(landmarks)
    print("Complete depth fusion with Kalman filter and map is ready to use!!")