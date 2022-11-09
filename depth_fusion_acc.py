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

# geometry
# def makeQ(depth, baseline, K):
#     bf = K[0, 0] * baseline
#     sigma_d = 0.5
#     sigma_z = sigma_d * depth**2/bf
#     sigma_z2 = sigma_z**2
#     sigma2 = np.array([0.25, 0.25, sigma_z2])
#     Q = np.diag(sigma2)
#     return Q
#
# def computeU(u, v ,depth, K, baseline, pose):
#     J = makeJacobian(u, v, depth, K)
#     Jt = np.transpose(J)
#     Q = makeQ(depth, baseline, K)
#     Rcw = pose[0]
#     Rwc = np.transpose(Rcw)
#
#     U_obs = np.linalg.multi_dot([Rwc, J, Q, Jt, Rcw])
#
#     return U_obs

# learning-base uncertainty
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

def compute_observation(v, u, K, sigma_z, pose, depth_obs):
    X_obs = iproj(pose, K, u, v, depth_obs)
    U_obs = computeU(u, v, depth_obs, K, sigma_z, pose)
    trace_map[v, u] = np.trace(U_obs)
    return (X_obs, U_obs, u, v)

def call_back_method(res):
    global cur_map, cur_uncertainties, points_new_counter, trace_map
    points_new_counter += 1
    cur_map.append(res[0])
    cur_uncertainties.append(res[1])
    trace_map[res[3], res[2]] = np.trace(res[1])

def kf_fusion(X_prior, X_obs, U_prior, U_obs):
    # U_post = (inv(U_obs) + inv(X_obs))
    # U_post = inv(U_post)
    Si = U_prior + U_obs
    Wi = U_prior.dot(inv(Si)) # kalman gain
    ei = X_obs - X_prior
    #update
    X_post = X_prior + Wi.dot(ei)
    U_post = U_prior - Wi.dot(U_prior)
    return X_post, U_post

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

    max_trace_value = 0
    boundbox_path = args.bbox_path
    max_depth = 30 #K[0, 0] * baseline
    start_ID = 179 #178+85
    n_frames = len(pipeline_depth_map_names)

    log_file = os.path.join(args.output_path, "KF_fused_maps/statistics/counter.txt")
    with open(log_file, 'w') as f:
        sys.stdout = f
        for i in tqdm(range(start_ID, n_frames), desc="Start to depth fusion & a build map", unit="depth maps"):
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
            # filter out moving object
            boundingboxs = check_rigid(boundbox_path, i)
            # isMoving = np.zeros((height, width), dtype=bool)
            for (lower_x, upper_x, lower_y, upper_y) in boundingboxs:
                # isMoving[lower_y:upper_y + 1, lower_x:upper_x + 1] = True
                pipeline_depth_map[lower_y:upper_y + 1, lower_x:upper_x + 1] = 0.0

            # load pose for current frame
            pose = img_name_SE3[frame_id]
            height, width = shape[0], shape[1]

            # load uncertainty map for Z(depth) value
            uncertainty_name = "uncert_" + frame_id + ".pfm"
            pipeline_uncertainty_path = os.path.join(args.uncertainty_path, uncertainty_name)
            z_uncertainty_map, _ = util_io.read_pfm(pipeline_uncertainty_path, isCentimeter=True)

            # # load confidence
            # conf_name = "conf_" + frame_id + ".pfm"
            # pipeline_conf_dir = "/home/wangweihan/Documents/my_project/cvpr2023/dataset/Scene06/conf/fog/"
            # pipeline_conf_path = os.path.join(pipeline_conf_dir, conf_name)
            # conf_map, _ = util_io.read_pfm(pipeline_conf_path, isCentimeter=False)

            # save trace
            trace_map = np.zeros((height, width))

            cur_map = []
            cur_uncertainties = []
            points_fused_counter = 0
            points_new_counter = 0
            visited = np.zeros((height, width), dtype=bool)

            if i == start_ID:
                pool = mp.Pool(mp.cpu_count())
                for v in range(0, height):
                    for u in range(0, width):
                        depth_obs = np.float64(pipeline_depth_map[v, u])
                        if depth_obs <= 0:
                            continue
                        z_uncertainty = z_uncertainty_map[v, u]
                        pool.apply_async(compute_observation, args=(v, u, K, z_uncertainty, pose, depth_obs),
                                         callback=call_back_method)
                        # pool.apply_async(compute_observation, args=(v, u, K, baseline, pose, depth_obs),
                        #                                   callback=call_back_method)

                pool.close()
                pool.join()
            elif i > start_ID:
                for (id, u_proj, v_proj, depth_prior) in projFromMap(prev_map, pose, K):
                    # out of range of image
                    if u_proj < 0 or u_proj >= width or v_proj < 0 or v_proj >= height:
                        continue

                    depth_obs = np.float64(pipeline_depth_map[v_proj, u_proj])
                    # invalid depth
                    if depth_obs <= 0 or visited[v_proj, u_proj]:
                        continue

                    # obs
                    X_obs = iproj(pose, K, u_proj, v_proj, depth_obs)
                    z_uncertainty = z_uncertainty_map[v, u]
                    U_obs = computeU(u_proj, v_proj, depth_obs, K, z_uncertainty, pose)
                    #prior
                    X_prior = prev_map[id]
                    U_prior = prev_uncertainties[id]
                    dist = np.linalg.norm(X_obs - X_prior)
                    th = 0.6  # change to arguement
                    if dist > th:
                        continue

                    if np.trace(U_obs) > np.trace(U_prior):
                        continue

                    # Update by Kalman filter
                    visited[v_proj, u_proj] = True
                    points_fused_counter += 1

                    X_post, U_post = kf_fusion(X_prior, X_obs, U_prior, U_obs)
                    depth_post = getDepth(pose, X_post)
                    pipeline_depth_map[v_proj, u_proj] = depth_post

                    # update 3D position and uncertainty
                    X_post = iproj(pose, K, u_proj, v_proj, depth_post)
                    cur_map.append(X_post)
                    cur_uncertainties.append(U_post)

                # add rest of pixel's 3D position into map
                pool = mp.Pool(mp.cpu_count())
                for v in range(0, height):
                    for u in range(0, width):
                        depth_obs = pipeline_depth_map[v, u]
                        if visited[v, u] or depth_obs <= 0:
                            continue
                        z_uncertainty = z_uncertainty_map[v, u]
                        pool.apply_async(compute_observation, args=(v, u, K, z_uncertainty, pose, depth_obs),
                                         callback=call_back_method)
                        # pool.apply_async(compute_observation, args=(v, u, K, baseline, pose, depth_obs),
                        #                  callback=call_back_method)
                pool.close()
                pool.join()

            prev_map = cur_map
            prev_uncertainties = cur_uncertainties

            # save KF_fused depth map
            fused_depth_map_name = "fused_depth_" + frame_id + ".pfm"
            save_fused_depth_map_path = os.path.join(args.output_path, fused_depth_map_name)
            util_io.write_pfm(save_fused_depth_map_path, pipeline_depth_map)
            # save trace
            trace_map_file = "KF_fused_maps/trace/trace_" + frame_id + ".pfm"
            save_trace_map_path = os.path.join(args.output_path, trace_map_file)
            util_io.write_pfm(save_trace_map_path, trace_map)

            if np.amax(trace_map) > max_trace_value:
                max_trace_value = np.amax(trace_map)
            print("{0}: There are {1} points are fused and {2} points are new".format(i, points_fused_counter, points_new_counter))
        print("max trace: {0}".format(max_trace_value))
        f.close()

    # Evaluate depth maps
    util_io.compare_depth_maps(start_ID, args, pipeline_depth_map_names, max_depth, "kf_fusion")
    util_io.compare_depth_maps(start_ID, args, pipeline_depth_map_names, max_depth)
    # conf depth_map
    util_io.compare_depth_maps(start_ID, args, pipeline_depth_map_names, max_depth, "conf_fusion")


