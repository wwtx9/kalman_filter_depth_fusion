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
import numba as nu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--src_folder",
                        help="path to pipeline's depth maps", type=str, default="")
    parser.add_argument("-p", "--src_poses_path",
                        help="path to pipeline's poses", type=str, default="extrinsic.txt")
    parser.add_argument("-k", "--intrinsic_path",
                        help="intrinsic_path path", type=str, default="extrinsic.txt")
    parser.add_argument("-c", "--uncertainty_path",
                        help="uncertainty ", type=str, default="")
    parser.add_argument("-t", "--gt_folder",
                        help="path to ground truth depth map", type=str)
    parser.add_argument("-o", "--output_path", default="./evaluation", type=str,
                        help="Output path where all metrics and results will be stored.")
    args = parser.parse_args()
    return args

# load poses from underwater dataset
# def load_poses(path):
#     f = open(path, "r")
#     lines = f.readlines()
#     img_name_poses_map = {}
#
#     for line in lines:
#         tokens = line.split(" ")
#         timestamp = tokens[0]
#         timestamp_tokens = timestamp.split(".")
#         img_name = timestamp_tokens[0]+timestamp_tokens[1]
#         twc = np.array([np.float64(tokens[1]), np.float64(tokens[2]), np.float64(tokens[3])])
#         qwc = np.quaternion(np.float64(tokens[7]), np.float64(tokens[4]), np.float64(tokens[5]), np.float64(tokens[6]))
#         Rwc = quaternion.as_rotation_matrix(qwc)
#         Rcw = np.transpose(Rwc)
#         tcw = -Rcw.dot(twc)
#         img_name_poses_map[img_name] = (Rcw, tcw)
#     f.close()
#     return img_name_poses_map

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
def projFromMap(map, pose, K): #map save 3D points with world coordinate
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    for i in range(len(map)):
        Rcw, tcw = pose[0], pose[1]
        landmark = map[i]
        Xc = Rcw.dot(landmark) + tcw

        u_proj = fx * Xc[0] / Xc[2] + cx
        v_proj = fy * Xc[1] / Xc[2] + cy
        yield i, round(u_proj), round(v_proj), Xc[2]

def getDepth(pose, K, Xw):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
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

def makeQ(depth, baseline, K):
    bf = K[0, 0] * baseline
    sigma_d = 0.5
    sigma_z = sigma_d * depth**2/bf
    sigma_z2 = sigma_z**2
    sigma2 = np.array([0.25, 0.25, sigma_z2])
    Q = np.diag(sigma2)
    return Q

def computeU(u, v ,depth, K, baseline, pose):
    J = makeJacobian(u, v, depth, K)
    Jt = np.transpose(J)
    Q = makeQ(depth, baseline, K)
    Rcw = pose[0]
    Rwc = np.transpose(Rcw)

    U_obs = np.linalg.multi_dot([Rwc, J, Q, Jt, Rcw])

    return U_obs

def read_pfm_depth_maps(path, isKitti2):
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
    if isKitti2:
        data /= 100
    return data, shape

def write_depth_map(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
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

def compare_depth_maps(tgt_folder, depth_map_folder, pipeline_depth_map_names, flag = "raw"):
    if flag == "raw":
        stats_file = os.path.join(args.output_path, "evaluation_raw_depth_maps.txt")
    else:
        stats_file = os.path.join(args.output_path, "evaluation_fused_depth_maps.txt")

    global_min_mae = 100.0
    global_max_mae = 0.0
    maes = []
    medians = []
    num_pixels = []
    total_both_no_zeros_pixels = 0

    for i in tqdm(range(len(pipeline_depth_map_names)), desc="Loading and comparing raw depth maps", unit="depth maps"):
        depth_name = pipeline_depth_map_names[i]
        gt_depth_name = "gt_" + depth_name
        gt_depth_map_path = tgt_folder + gt_depth_name + '.pfm'
        # depth maps
        gt_depth_map, _ = read_pfm_depth_maps(gt_depth_map_path, isKitti2=True)

        pipline_depth_map_path = depth_map_folder + pipeline_depth_map_names[i] + '.pfm'
        pipeline_depth_map, _ = read_pfm_depth_maps(pipline_depth_map_path, isKitti2=True)

        both_non_zeros = (pipeline_depth_map != 0) & (gt_depth_map != 0)
        both_zeros = (pipeline_depth_map == 0) & (gt_depth_map == 0)
        if np.sum(both_non_zeros) == 0:
            continue
        median = np.median(np.absolute(pipeline_depth_map[both_non_zeros] - gt_depth_map[both_non_zeros]))
        medians.append(median)
        mae = np.mean(np.absolute(pipeline_depth_map[both_non_zeros] - gt_depth_map[both_non_zeros]))
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

if __name__ == "__main__":
    args = parse_args()
    print(args)

    K = load_intrinsic(args.intrinsic_path)
    baseline = 0.532725 # in meter

    regex_exp = re.compile(r'pfm')
    pipeline_depth_map_paths = [os.path.join(args.src_folder, f) for f in os.listdir(args.src_folder) if regex_exp.search(f)]
    pipeline_depth_map_paths.sort()

    patten = args.src_folder + '(.+?).pfm'
    pipeline_depth_map_names_2d = [re.findall(patten, depth_map_path) for depth_map_path in pipeline_depth_map_paths]
    pipeline_depth_map_names = [i for iterm in pipeline_depth_map_names_2d for i in iterm]
    pipeline_depth_map_names.sort()

    img_name_SE3 = load_poses(args.src_poses_path, isLeft=True)
    assert (len(img_name_SE3) == len(pipeline_depth_map_names))

    prev_map = []
    prev_uncertainties = []

    pipeline_depth_map_names = pipeline_depth_map_names[0:5]
    for i in tqdm(range(0, len(pipeline_depth_map_names)), desc="Start to depth fusion & a build map", unit="depth maps"):
        pipline_depth_map_path = pipeline_depth_map_paths[i]
        pipeline_depth_map, shape = read_pfm_depth_maps(pipline_depth_map_path, isKitti2=True)
        frame_id = pipeline_depth_map_names[i][6:]
        pose = img_name_SE3[frame_id]

        height, width = shape[0], shape[1]
        if i == 0:
            for v in range(0, height):
                for u in range(0, width):
                    depth = pipeline_depth_map[v, u]
                    if depth <= 0:
                        continue
                    X_obs = iproj(pose, K, u, v, depth)
                    prev_map.append(X_obs)
                    U_obs = computeU(u, v, depth, K, baseline, pose)
                    prev_uncertainties.append(U_obs)
        else:
            cur_map = []
            cur_uncertainties = []
            visited = np.zeros((height, width), dtype=bool)
            for (id, u_proj, v_proj, depth_prior) in projFromMap(prev_map, pose, K):
                if u_proj < 0 or u_proj >= width or v_proj < 0 or v_proj >= height:
                    continue
                depth_obs = pipeline_depth_map[v_proj, u_proj]
                if depth_obs <= 0 or visited[v_proj, u_proj]:
                    continue


                # bf = K[0, 0] * baseline
                # sigma_d = 0.5
                # th = sigma_d * depth_prior**2/bf
                # if abs(depth_obs - depth_prior) > th:
                #     continue

                X_obs = iproj(pose, K, u_proj, v_proj, depth_obs)
                U_obs = computeU(u_proj, v_proj, depth_obs, K, baseline, pose)

                U_prior = prev_uncertainties[id]
                X_prior = prev_map[id]

                lamda = np.linalg.eig(U_prior)
                index = np.argmax(lamda[0])
                max_eigenval = np.real(lamda[0][index])

                dist = np.linalg.norm(X_obs-X_prior)
                if dist > np.sqrt(max_eigenval):
                    continue

                visited[v_proj, u_proj] = True
                # Update by Kalman filter
                X_post, U_post = kf_depth_fusion(X_prior, X_obs, U_prior, U_obs)
                depth_post = getDepth(pose, K, X_post)

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

                    X_obs = iproj(pose, K, u, v, depth_obs)
                    U_obs = computeU(u, v, depth_obs, K, baseline, pose)
                    cur_map.append(X_obs)
                    cur_uncertainties.append(U_obs)

            prev_map = cur_map
            prev_uncertainties = cur_uncertainties

        save_path = args.output_path + "kf_depth_maps/" + pipeline_depth_map_names[i] + ".pfm"
        write_depth_map(pipeline_depth_map, save_path)

    # # print("Complete depth map {0}:{1}".format(i, pipeline_depth_map_names[i]))

    # Evaluate depth maps
    compare_depth_maps(args.gt_folder, args.src_folder, pipeline_depth_map_names)
    # fused_depth_folder = args.output_path + "kf_depth_maps/"
    # compare_depth_maps(args.gt_folder, fused_depth_folder, pipeline_depth_map_names, "fused")

    # Pass global_map to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # landmarks = np.asarray(global_map)
    # pcd.points = o3d.utility.Vector3dVector(landmarks)
    print("Complete depth fusion with Kalman filter and map is ready to use!!")