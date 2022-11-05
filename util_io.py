#!/usr/bin/env python

import numpy as np
import re
import sys

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

def read_pfm(path, isCentimeter=False):
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
    if isCentimeter:
        data /= 100
    return data, shape

def write_depth_map(fpath, data, scale=1, file_identifier=b'Pf', dtype="float32"):
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
