#!/usr/bin/env python
import argparse
from plistlib import FMT_XML
import numpy as np
import os
import re
import pylab as plt
import sys
from tqdm import tqdm
import open3d as o3d
import quaternion
from numpy.linalg import inv
from numba import jit
import cv2
import util_io
import multiprocessing as mp
def check_rigid(path, curFrameId):
    f = open(path, "r")
    lines = f.readlines()
    lines = lines[1:]
    bondingbox = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        tokens = line.split(" ")
        frameId = int(tokens[0])
        cameraId = tokens[1]
        isMoving = tokens[10]
        if frameId < curFrameId:
            continue
        elif frameId == curFrameId and cameraId == str(0) and isMoving == "True":
            left, right, top, bottom = int(tokens[3]), int(tokens[4]), int(tokens[5]), int(tokens[6])
            # lefttop = (left, top) #x,y
            # rightbottom = (right, bottom) #x,y
            lower_x, upper_x, lower_y, upper_y = left, right, top, bottom
            bondingbox.append((lower_x, upper_x, lower_y, upper_y))
        else:
            return bondingbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--src_folder",
                        help="path to pipeline's depth maps", type=str, default="")
    parser.add_argument("-p", "--src_poses_path",
                        help="path to pipeline's poses", type=str, default="extrinsic.txt")
    parser.add_argument("-k", "--intrinsic_path",
                        help="intrinsic_path path", type=str, default="extrinsic.txt")
    parser.add_argument("-u", "--uncertainty_path",
                        help="uncertainty ", type=str, default="")
    parser.add_argument("-m", "--mask_path",
                        help="path to mask ", type=str, default="")
    parser.add_argument("-b", "--bbox_path",
                        help="path to bbox.txt ", type=str, default="bbox.txt")
    parser.add_argument("-t", "--gt_folder",
                        help="path to ground truth depth map", type=str)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                        help="Output path where all metrics and results will be stored.")
    args = parser.parse_args()
    return args

def projFromMap(map, pose, K): #map save 3D points with world coordinate
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    for i in range(len(map)):
        Rcw, tcw = pose[0], pose[1]
        landmark = map[i]
        Xc = Rcw.dot(landmark) + tcw

        u_proj = fx * Xc[0] / Xc[2] + cx
        v_proj = fy * Xc[1] / Xc[2] + cy
        yield i, round(u_proj), round(v_proj), Xc[2]

def getDepth(pose, Xw):
    Rcw, tcw = pose[0], pose[1]

    Xc = Rcw.dot(Xw) + tcw
    return Xc[2]

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
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.array([[Z/fx, 0, (u-cx)/fx],
                  [0, Z/fy, (v-cy)/fy],
                  [0, 0, 1]])
    return J

def makeQ(sigma_z):
    sigma_z2 = sigma_z**2
    sigma2 = np.array([0.25, 0.25, sigma_z2])
    Q = np.diag(sigma2)
    return Q

def computeU(u, v ,depth, K, sigma_z, pose):
    J = makeJacobian(u, v, depth, K)
    Jt = np.transpose(J)
    Q = makeQ(sigma_z)
    Rcw = pose[0]
    Rwc = np.transpose(Rcw)
    U_obs = np.linalg.multi_dot([Rwc, J, Q, Jt, Rcw])
    return U_obs

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

def compare_depth_maps(start_ID, args, pipeline_depth_map_names, points_fused_counters, points_new_counters, flag = "raw"):
    if flag == "raw":
        stats_file = os.path.join(args.output_path, "statistics/evaluation_raw_depth_maps.txt")
    else:
        stats_file = os.path.join(args.output_path, "statistics/evaluation_fused_depth_maps.txt")

    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_valid_pixels = 0
    max_depth = K[0, 0] * baseline
    n_frames = 182 #len(pipeline_depth_map_names)

    for i in tqdm(range(start_ID, n_frames), desc="Loading and comparing raw depth maps", unit="depth maps"):
        depth_name = pipeline_depth_map_names[i][-6:]
        gt_depth_name = "gt_depth_" + depth_name + ".pfm"
        gt_depth_map_path = os.path.join(args.gt_folder, gt_depth_name)
        # depth maps
        gt_depth_map, shape = util_io.read_pfm(gt_depth_map_path, isCentimeter=True)
        gt_depth_map[gt_depth_map >= max_depth] = 0

        height, width = shape[0], shape[1]

        if flag is "raw":
            raw_depth_name = "depth_" + depth_name + ".pfm"
            raw_depth_map_path = os.path.join(args.src_folder, raw_depth_name)
            pipeline_depth_map, _ = util_io.read_pfm(raw_depth_map_path, isCentimeter=True)

            # load mask
            frame_id = pipeline_depth_map_names[i][-6:]
            mask_name = "mask_" + frame_id + ".pfm"
            mask_path = os.path.join(args.mask_path, mask_name)
            mask_map, _ = util_io.read_pfm(mask_path, isCentimeter=False)

            # filter out maximum depth and mask
            pipeline_depth_map[pipeline_depth_map >= max_depth] = 0.0
            pipeline_depth_map[mask_map == 0.0] = 0.0

            # load bounding box
            boundingboxs = check_rigid(boundbox_path, i)

            for v in range(0, height):
                for u in range(0, width):
                    if any((lower1 <= u <= upper1) and (lower2 <= v <= upper2) for (lower1, upper1, lower2, upper2) in
                           boundingboxs):
                        pipeline_depth_map[v, u] = 0.0
                        continue
        else:
            fused_depth_name = "fused_depth_" + depth_name + ".pfm"
            fused_depth_map_path = os.path.join(args.output_path, fused_depth_name)
            pipeline_depth_map, _ = util_io.read_pfm(fused_depth_map_path, isCentimeter=False)

        both_valid = (pipeline_depth_map != 0) & (gt_depth_map != 0)
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
            f.write("No.{0} \n".format(i+start_ID))
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


if __name__ == "__main__":
    args = parse_args()
    print(args)

    K = util_io.load_intrinsic(args.intrinsic_path)
    baseline = 0.532725  # in meter

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if
                                regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()

    img_name_SE3 = util_io.load_poses(args.src_poses_path, isLeft=True)
    assert (len(img_name_SE3) == len(pipeline_depth_map_names))

    prev_map = []
    prev_uncertainties = []
    points_fused_counters = []
    points_new_counters = []
    # pipeline_depth_map_names = pipeline_depth_map_names[180:]

    boundbox_path = args.bbox_path
    max_depth = K[0, 0] * baseline
    start_ID = 179
    n_frames = 182 #len(pipeline_depth_map_names)
    for i in tqdm(range(start_ID, n_frames), desc="Start to depth fusion & a build map",
                  unit="depth maps"):
        # depth map path
        pipline_depth_map_path = pipeline_depth_map_paths[i]
        # load depth map
        pipeline_depth_map, shape = util_io.read_pfm(pipline_depth_map_path, isCentimeter=True)

        # load mask
        frame_id = pipeline_depth_map_names[i][-6:]
        mask_name = "mask_" + frame_id + ".pfm"
        mask_path = os.path.join(args.mask_path, mask_name)
        mask_map, _ = util_io.read_pfm(mask_path, isCentimeter=False)

        # filter out maximum depth and mask
        pipeline_depth_map[pipeline_depth_map >= max_depth] = 0.0
        pipeline_depth_map[mask_map == 0.0] = 0.0

        # load pose for current frame
        pose = img_name_SE3[frame_id]
        height, width = shape[0], shape[1]

        # load bounding box
        boundingboxs = check_rigid(boundbox_path, i)

        # load uncertainty map for Z(depth) value
        uncertainty_name = "uncert_" + frame_id + ".pfm"
        pipeline_uncertainty_path = os.path.join(args.uncertainty_path, uncertainty_name)
        z_uncertainty_map, _ = util_io.read_pfm(pipeline_uncertainty_path, isCentimeter=True)

        error_map = np.zeros((height, width))
        cur_map = []
        cur_uncertainties = []
        points_fused_counter = 0
        points_new_counter = 0

        if i == start_ID:
            for v in range(0, height):
                for u in range(0, width):
                    depth_obs = np.float64(pipeline_depth_map[v, u])
                    if depth_obs <= 0:
                        continue
                    if any((lower1 <= u <= upper1) and (lower2 <= v <= upper2) for (lower1, upper1, lower2, upper2) in
                           boundingboxs):
                        pipeline_depth_map[v, u] = 0.0
                        continue
                    points_new_counter += 1
                    X_obs = iproj(pose, K, u, v, depth_obs)
                    cur_map.append(X_obs)
                    z_uncertainty = z_uncertainty_map[v, u]
                    U_obs = computeU(u, v, depth_obs, K, z_uncertainty, pose)
                    # U_obs = computeU(u, v, depth, K, baseline, pose)
                    cur_uncertainties.append(U_obs)
        else:
            visited = np.zeros((height, width), dtype=bool)
            for (id, u_proj, v_proj, depth_prior) in projFromMap(prev_map, pose, K):
                # out of range of image
                if u_proj < 0 or u_proj >= width or v_proj < 0 or v_proj >= height:
                    continue
                # not rigid
                if any((lower1 <= u_proj <= upper1) and (lower2 <= v_proj <= upper2) for (lower1, upper1, lower2, upper2) in
                       boundingboxs):
                    pipeline_depth_map[v_proj, u_proj] = 0.0
                    continue

                depth_obs = np.float64(pipeline_depth_map[v_proj, u_proj])
                # invalid depth
                if depth_obs <= 0 or visited[v_proj, u_proj]:
                    continue

                X_obs = iproj(pose, K, u_proj, v_proj, depth_obs)
                z_uncertainty = z_uncertainty_map[v_proj, u_proj]
                U_obs = computeU(u_proj, v_proj, depth_obs, K, z_uncertainty, pose)
                # U_obs = computeU(u_proj, v_proj, depth_obs, K, baseline, pose)
                U_prior = prev_uncertainties[id]
                X_prior = prev_map[id]

                dist = np.linalg.norm(X_obs - X_prior)
                th = 0.1
                if dist > th:
                    continue

                visited[v_proj, u_proj] = True
                points_fused_counter += 1
                # Update by Kalman filter
                X_post, U_post = kf_depth_fusion(X_prior, X_obs, U_prior, U_obs)
                # error_map[v_proj, u_proj] = error
                depth_post = getDepth(pose, X_post)

                pipeline_depth_map[v_proj, u_proj] = depth_post
                # update 3D position and uncertainty
                cur_map.append(X_post)
                cur_uncertainties.append(U_post)

            # add rest of pixel's 3D position into map
            for v in range(0, height):
                for u in range(0, width):
                    depth_obs = pipeline_depth_map[v, u]
                    if visited[v, u] or depth_obs <= 0:
                        continue

                    if any((lower1 <= u <= upper1) and (lower2 <= v <= upper2) for
                           (lower1, upper1, lower2, upper2) in
                           boundingboxs):
                        pipeline_depth_map[v, u] = 0.0
                        continue

                    points_new_counter += 1
                    X_obs = iproj(pose, K, u, v, depth_obs)
                    z_uncertainty = z_uncertainty_map[v, u]
                    U_obs = computeU(u, v, depth_obs, K, z_uncertainty, pose)
                    # U_obs = computeU(u, v, depth_obs, K, baseline, pose)
                    cur_map.append(X_obs)
                    cur_uncertainties.append(U_obs)

        prev_map = cur_map
        prev_uncertainties = cur_uncertainties

        points_fused_counters.append(points_fused_counter)
        points_new_counters.append(points_new_counter)

        fused_depth_map_name = "fused_depth_" + frame_id + ".pfm"
        save_fused_depth_map_path = os.path.join(args.output_path, fused_depth_map_name)
        util_io.write_depth_map(save_fused_depth_map_path, pipeline_depth_map)

        # save error map
        # error_png = np.where(error_map >= 0.1, 255, 0).astype(np.uint8)
        # # error_png = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # save_path = args.output_path + "error_" + pipeline_depth_map_names[i] + ".png"
        # cv2.imwrite(save_path, error_png)
        # save_path = args.output_path + "error_" + pipeline_depth_map_names[i] + ".pfm"
        # util_io.write_depth_map(save_path, error_map)


    # Evaluate depth maps
    compare_depth_maps(start_ID, args, pipeline_depth_map_names, points_fused_counters,
                       points_new_counters)
    compare_depth_maps(start_ID, args, pipeline_depth_map_names, points_fused_counters,
                       points_new_counters, "fused")

    print("Complete depth fusion with Kalman filter and map is ready to use!!")
