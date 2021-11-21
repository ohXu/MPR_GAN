# coding=gbk
import numpy as np
from PIL import Image
import os
import cv2
import time
import math
import torch
from scipy.optimize import curve_fit
from collections import Counter
import matplotlib.pyplot as plt
import shutil

def first_clip_photo():
    for root, dirs, files in os.walk('../dataset/KITTI/image_2'):
        for file in files:
            (filename, extension) = os.path.splitext(file)
            img = Image.open(root + '/' + file)
            img = img.crop([620 - 384, img.size[1] - 192, 620 + 384, img.size[1]])
            img.save('../dataset/KITTI/image_crop/' + filename + '.png')


def second_choose_colorization_points():
    f = open('../dataset/KITTI/calib.txt', 'r')
    line = f.readline()
    i = 0
    p2 = tr = None
    while line:
        line = f.readline()
        a = line.strip().split(' ')
        i = i + 1
        if i == 2:
            p2 = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])], [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])]])
        if i == 4:
            tr = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])], [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])], [0, 0, 0, 1]])
    f.close()
    transformation_matrix = np.dot(p2, tr)
    for root, dirs, files in os.walk('../dataset/KITTI/image_crop'):
        for file in files:
            (filename, extension) = os.path.splitext(file)
            id = int(filename)
            points = np.fromfile('../dataset/KITTI/velodyne/' + str(id).zfill(6) + ".bin", dtype=np.float32,
                                 count=-1).reshape([-1, 4])
            x = points[:, 0]  # x position of point
            y = points[:, 1]  # y position of point
            z = points[:, 2]  # z position of point
            reflex = points[:, 3]

            img = cv2.imread('../dataset/KITTI/image_crop/' + str(id).zfill(6) + ".png")
            rgb_img = np.zeros([192, 768, 3], dtype='float64')
            rgb_img.fill(255)
            rgb_img_R = np.zeros([192, 768, 3], dtype='float64')
            rgb_img_R.fill(255)

            rgb_img2 = np.zeros([96, 384, 3], dtype='float64')
            rgb_img2.fill(255)
            rgb_img2_R = np.zeros([96, 384, 3], dtype='float64')
            rgb_img2_R.fill(255)

            rgb_img3 = np.zeros([64, 256, 3], dtype='float64')
            # rgb_img3.fill(255)
            rgb_img3_R = np.zeros([64, 256, 3], dtype='float64')
            # rgb_img3_R.fill(255)

            rgb_img4 = np.zeros([48, 192, 3], dtype='float64')
            rgb_img4.fill(255)
            rgb_img4_R = np.zeros([48, 192, 3], dtype='float64')
            rgb_img4_R.fill(255)

            depth_img = np.zeros([192, 768], dtype='float64')
            depth_img.fill(1000000)

            depth_img2 = np.zeros([96, 384], dtype='float64')
            depth_img2.fill(1000000)

            depth_img4 = np.zeros([48, 192], dtype='float64')
            depth_img4.fill(1000000)

            depth_img3 = np.zeros([64, 256], dtype='float64')
            depth_img3.fill(1000000)

            scan = []
            fw = open('../dataset/KITTI/pc_rgb/' + str(id).zfill(6) + ".txt", 'w')
            for i in range(points.shape[0]):
                if x[i] >= 0:
                    p = np.array([[x[i]], [y[i]], [z[i]], [1]])
                    p = np.dot(transformation_matrix, p)
                    u = p[0, 0] / p[2, 0] - 236.0
                    v = p[1, 0] / p[2, 0] - 184.0

                    u2 = (p[0, 0] / p[2, 0] - 236.0) / 2.0
                    v2 = (p[1, 0] / p[2, 0] - 184.0) / 2.0

                    u3 = (p[0, 0] / p[2, 0] - 236.0) / 3.0
                    v3 = (p[1, 0] / p[2, 0] - 184.0) / 3.0

                    u4 = (p[0, 0] / p[2, 0] - 236.0) / 4.0
                    v4 = (p[1, 0] / p[2, 0] - 184.0) / 4.0

                    if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                        r = img[int(v), int(u)][2]
                        g = img[int(v), int(u)][1]
                        b = img[int(v), int(u)][0]
                        scan.append([x[i], y[i], z[i], reflex[i], r, g, b, v, u, v2, u2, v3, u3, v4, u4])

                        fw.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + " " + str(r) + " " + str(g) + " " + str(b))
                        fw.write("\n")
            fw.close()

            scan = np.array(scan)
            # scan = scan[np.argsort(scan[:, -1])]

            points = scan[:, 0:3]
            remission = scan[:, 3]
            colors = scan[:, 4:7]
            depth = np.linalg.norm(points, 2, axis=1)
            scan_x = points[:, 0]
            scan_y = points[:, 1]
            scan_z = points[:, 2]

            # max_depth = np.max(depth)
            # min_depth = np.min(depth)
            # depth = (depth - min_depth) / (max_depth - min_depth) * 255.0
            # max_remission = np.max(remission)
            # min_remission = np.min(remission)
            # remission = (remission - min_remission) / (max_remission - min_remission) * 255.0
            # max_z = np.max(scan_z)
            # min_z = np.min(scan_z)
            # scan_z = (scan_z - min_z) / (max_z - min_z) * 255.0

            allnp = np.array(depth)
            allnp = np.sort(allnp)
            allnp = np.around(allnp, 0)
            c = Counter(allnp.flatten())
            labels, values = zip(*c.items())
            labels = np.array(list(labels))
            values = np.array(list(values))
            values = values / allnp.shape[0]
            for i in range(1, values.shape[0]):
                values[i] = values[i] + values[i - 1]
            para, pcov = curve_fit(fun_five, labels, values)
            depth = fun_five(depth, para[0], para[1], para[2], para[3], para[4], para[5]) * 255.0
            remission = remission * 255.0

            znp = np.array(scan_z)
            znp = np.sort(znp)
            znp = np.around(znp, 1)
            c = Counter(znp.flatten())
            zlabels, zvalues = zip(*c.items())
            zlabels = np.array(list(zlabels))
            zvalues = np.array(list(zvalues))
            zvalues = zvalues / znp.shape[0]
            for i in range(1, zvalues.shape[0]):
                zvalues[i] = zvalues[i] + zvalues[i - 1]
            zpara, zpcov = curve_fit(fun_six, zlabels, zvalues)
            scan_z = fun_six(scan_z, zpara[0], zpara[1], zpara[2], zpara[3], zpara[4], zpara[5], zpara[6]) * 255.0

            for i in range(scan.shape[0]):
                if points[i, 0] >= 0:

                    u = scan[i, 8]
                    v = scan[i, 7]

                    u2 = scan[i, 10]
                    v2 = scan[i, 9]

                    u3 = scan[i, 12]
                    v3 = scan[i, 11]

                    u4 = scan[i, 14]
                    v4 = scan[i, 13]

                    if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                        if depth_img[int(v), int(u)] > depth[i]:
                            depth_img[int(v), int(u)] = depth[i]
                            rgb_img[int(v), int(u)][0] = depth[i]
                            rgb_img[int(v), int(u)][1] = remission[i]
                            rgb_img[int(v), int(u)][2] = scan_z[i]

                            rgb_img_R[int(v), int(u)][0] = colors[i, 0]
                            rgb_img_R[int(v), int(u)][1] = colors[i, 1]
                            rgb_img_R[int(v), int(u)][2] = colors[i, 2]

                        if depth_img2[int(v2), int(u2)] > depth[i]:
                            depth_img2[int(v2), int(u2)] = depth[i]
                            rgb_img2[int(v2), int(u2)][0] = depth[i]
                            rgb_img2[int(v2), int(u2)][1] = remission[i]
                            rgb_img2[int(v2), int(u2)][2] = scan_z[i]

                            rgb_img2_R[int(v2), int(u2)][0] = colors[i, 0]
                            rgb_img2_R[int(v2), int(u2)][1] = colors[i, 1]
                            rgb_img2_R[int(v2), int(u2)][2] = colors[i, 2]

                        if depth_img3[int(v3), int(u3)] > depth[i]:
                            depth_img3[int(v3), int(u3)] = depth[i]
                            rgb_img3[int(v3), int(u3)][0] = depth[i]
                            rgb_img3[int(v3), int(u3)][1] = remission[i]
                            rgb_img3[int(v3), int(u3)][2] = scan_z[i]

                            rgb_img3_R[int(v3), int(u3)][0] = colors[i, 0]
                            rgb_img3_R[int(v3), int(u3)][1] = colors[i, 1]
                            rgb_img3_R[int(v3), int(u3)][2] = colors[i, 2]

                        if depth_img4[int(v4), int(u4)] > depth[i]:
                            depth_img4[int(v4), int(u4)] = depth[i]
                            rgb_img4[int(v4), int(u4)][0] = depth[i]
                            rgb_img4[int(v4), int(u4)][1] = remission[i]
                            rgb_img4[int(v4), int(u4)][2] = scan_z[i]
                            rgb_img4_R[int(v4), int(u4)][0] = colors[i, 0]
                            rgb_img4_R[int(v4), int(u4)][1] = colors[i, 1]
                            rgb_img4_R[int(v4), int(u4)][2] = colors[i, 2]

            rgb_img3 = Image.fromarray(np.uint8(rgb_img3))
            rgb_img3.save('../dataset/KITTI/2D_projected_image/' + str(id) + '.png')
            print(id)


def third_assign_points():

    for root, dirs, files in os.walk('../dataset/KITTI/2D_projected_image'):
        for file in files:
            (filename, extension) = os.path.splitext(file)

            src = '../dataset/KITTI/2D_projected_image/' + filename + '.png'
            dst = '../dataset/KITTI/test_A/' + filename + '.png'
            shutil.copy(src, dst)
            src = '../dataset/KITTI/image_crop/' + filename.zfill(6) + '.png'
            dst = '../dataset/KITTI/test_B/' + filename + '.png'
            shutil.copy(src, dst)
            if int(filename) % 5 != 0:
                src = '../dataset/KITTI/test_A/' + filename + '.png'
                dst = '../dataset/KITTI/train_A/' + filename + '.png'
                shutil.copy(src, dst)
                src = '../dataset/KITTI/test_B/' + filename + '.png'
                dst = '../dataset/KITTI/train_B/' + filename + '.png'
                shutil.copy(src, dst)


def fifth_voxel_1_1():
    for root, dirs, files in os.walk('../dataset/KITTI/pc_rgb'):
        i = 0
        for file in files:
            scan = np.loadtxt('../dataset/KITTI/pc_rgb/' + file,
                              dtype=np.float32)
            u, v, d, index, \
            groups_each, groups_index, groups_each_index, split_each_begin, split_each_begin_in_group, split_each_max, \
            distance = fifth_voxel_2_1(scan)

            np.savez_compressed('../dataset/KITTI/voxel/%s_compressed' % i, u=u, v=v, d=d,
                                select_index=index,
                                group_belongs=groups_each, index_in_each_group=groups_each_index,
                                distance=distance, each_split_max_num=split_each_max)
            i = i + 1

def fifth_voxel_2_1(scan):
    num_planes = 1
    proj_W = 256
    proj_H = 64

    points = scan[:, 0:3]
    depth_all = np.linalg.norm(points, 2, axis=1)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    f = open('../dataset/KITTI/calib.txt', 'r')
    line = f.readline()
    i = 0
    p2 = tr = None
    while line:
        line = f.readline()
        a = line.strip().split(' ')
        i = i + 1
        if i == 2:
            p2 = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])], [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])]])
        if i == 4:
            tr = np.array(
                [[eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4])], [eval(a[5]), eval(a[6]), eval(a[7]), eval(a[8])],
                 [eval(a[9]), eval(a[10]), eval(a[11]), eval(a[12])], [0, 0, 0, 1]])
    f.close()
    transformation_matrix = np.dot(p2, tr)
    st = time.time()
    u_all = []
    v_all = []
    for i1 in range(points.shape[0]):
        if x[i1] >= 0:
            p = np.array([[x[i1]], [y[i1]], [z[i1]], [1]])
            p = np.dot(transformation_matrix, p)
            u = p[0, 0] / p[2, 0] - 236.0
            v = p[1, 0] / p[2, 0] - 184.0
            u_all.append(int(u / 3.0))
            v_all.append(int(v / 3.0))
    # yaw = -np.arctan2(scan_y, scan_x)
    # pitch = np.arcsin(scan_z / depth_all)
    #
    # fov_up = max(pitch)
    # fov_down = min(pitch)
    # fov_left = max(yaw)
    # fov_right = min(yaw)
    # fov = abs(fov_up - fov_down)
    # fov2 = abs(fov_right - fov_left)
    #
    # proj_x = (yaw - fov_right) / fov2
    # proj_y = (fov_up - pitch) / fov
    #
    # u_all = proj_W * proj_x  # in [0.0, W]
    # v_all = proj_H * proj_y  # in [0.0, H]
    #
    # u_all = np.floor(u_all)
    # u_all = np.minimum(proj_W - 1, u_all)
    # u_all = np.maximum(0, u_all).astype(np.int32)
    # v_all = np.floor(v_all)
    # v_all = np.minimum(proj_H - 1, v_all)
    # v_all = np.maximum(0, v_all).astype(np.int32)

    u_all = np.array(u_all)
    v_all = np.array(v_all)
    valid_u = np.where((u_all >= 0) & (u_all <= (proj_W - 1)))
    valid_v = np.where((v_all >= 0) & (v_all <= (proj_H - 1)))
    valid_d = np.where((depth_all > 0) & (depth_all < 1000.0))
    print(len(valid_u[0]), len(valid_v[0]), len(valid_d[0]))

    valid_position = np.intersect1d(valid_u, valid_v)
    valid_position = np.intersect1d(valid_position, valid_d)
    selected_depth = depth_all[valid_position]
    index = np.argsort(-selected_depth)

    valid_position_sorted = valid_position[index]
    valid_d_sorted = depth_all[valid_position_sorted]
    center_u_sorted = u_all[valid_position_sorted]
    center_v_soretd = v_all[valid_position_sorted]

    u_sorted = np.uint32(np.rint(center_u_sorted))
    v_sorted = np.uint32(np.rint(center_v_soretd))
    distance_sorted = np.sqrt(np.square(u_sorted - center_u_sorted) + np.square(v_sorted - center_v_soretd))
    num_valids = len(index)

    valid_d_min = valid_d_sorted[num_valids - 1]
    valid_d_max = valid_d_sorted[0]
    tmp = np.linspace(valid_d_max, valid_d_min, num_planes + 1)
    up_boundary = tmp[1:]

    d_position = np.zeros([num_valids])
    print('有效', num_valids, tmp)

    cnt = 0
    for i in range(num_valids):
        tmp_d = valid_d_sorted[i]
        if tmp_d >= up_boundary[cnt]:
            d_position[i] = num_planes - cnt - 1
        else:
            for j in range(1, num_planes - cnt):
                cnt = cnt + 1
                if tmp_d >= up_boundary[cnt]:
                    d_position[i] = num_planes - cnt - 1
                    break

    groups_original = u_sorted + v_sorted * proj_W + d_position * proj_W * proj_H  # groups
    groups_original_sort_index = np.argsort(groups_original)  # small to large

    groups_original_sorted = groups_original[groups_original_sort_index]
    u_sorted_1 = u_sorted[groups_original_sort_index]
    v_sorted_1 = v_sorted[groups_original_sort_index]
    d_position_sorted_1 = d_position[groups_original_sort_index]
    valid_position_sorted_1 = valid_position_sorted[groups_original_sort_index]
    distance_sorted_1 = distance_sorted[groups_original_sort_index]

    array = np.uint16(np.linspace(0, 5000, 5000,
                                  endpoint=False))  # assign points within one voxel or group a sequence index. Begin from 0. The max num in each group less than 1000.
    groups_index = np.zeros_like(valid_position_sorted_1)  # each group's start position.
    groups_each = np.zeros_like(valid_position_sorted_1)  # each point belongs to which group or voxel.
    groups_each_index = np.zeros_like(valid_position_sorted_1,
                                      dtype=np.uint16)  # each point's index/order in one group, a sequence.

    group_begin = 0
    cnt = 0

    for ii in range(num_valids):
        group_tmp = groups_original_sorted[ii]
        if (ii + 1) < num_valids:
            group_next = groups_original_sorted[ii + 1]
            if not group_tmp == group_next:
                groups_each[group_begin:(ii + 1)] = cnt
                groups_each_index[group_begin:(ii + 1)] = array[0:(ii + 1 - group_begin)]
                groups_index[cnt] = group_begin
                cnt = cnt + 1
                group_begin = ii + 1
            # else:
            #     print(group_tmp, group_next)
        else:
            groups_each[group_begin:] = cnt
            groups_each_index[group_begin:] = array[0:(num_valids - group_begin)]
            groups_index[cnt] = group_begin

    groups_index = groups_index[0:(cnt + 1)]
    split_each_max = np.zeros(num_planes, dtype=np.uint16)

    split_position = np.where((d_position_sorted_1[groups_index] - np.concatenate(
        (np.array([0]),
         d_position_sorted_1[groups_index][0:-1]))) > 0)  # find split position of different planes.
    split_each_begin = np.concatenate((np.array([0]), groups_index[split_position]))
    split_each_begin_in_group = np.concatenate((np.array([0]), split_position[0]))

    d_valid = d_position_sorted_1[groups_index[split_each_begin_in_group]]

    for j in range(len(split_each_begin)):

        begin = split_each_begin[j]
        if j < (len(split_each_begin_in_group) - 1):
            end = split_each_begin[j + 1]
            max_num = np.max(groups_each_index[begin:end]) + 1
            split_each_max[int(d_valid[j])] = max_num
        else:
            max_num = np.max(groups_each_index[begin:]) + 1
            split_each_max[int(d_valid[j])] = max_num
    print('split_time: %s' % (time.time() - st))
    return np.uint16(u_sorted_1), np.uint16(v_sorted_1), np.uint8(d_position_sorted_1), np.uint32(
        valid_position_sorted_1), \
           np.uint32(groups_each), np.uint32(groups_index), np.uint16(groups_each_index), \
           np.uint32(split_each_begin), np.uint32(split_each_begin_in_group), np.uint16(split_each_max), \
           np.float16(distance_sorted_1)


def sixth_aggregation_1_1():
    for root, dirs, files in os.walk('../dataset/KITTI/pc_rgb'):
        i = 0
        for file in files:
            scan = np.loadtxt('../dataset/KITTI/pc_rgb/' + file,
                              dtype=np.float32)
            npzfile = np.load('../dataset/KITTI/voxel/%s_compressed.npz' % i)
            weight_average, distance_to_depth_min = sixth_aggregation_2_1(scan, npzfile, 1, 1)
            np.savez_compressed('../dataset/KITTI/aggregation/' + '%s_weight' % i,
                                weight_average=weight_average,
                                distance_to_depth_min=distance_to_depth_min)
            i += 1


def sixth_aggregation_2_1(scan, npzfile, a, b):
    select_index = npzfile['select_index']  # select_index begin from 0.   index
    index_in_each_group = npzfile['index_in_each_group']
    distance = npzfile['distance']
    start = time.time()
    points = scan[:, 0:3]
    depth_all = np.linalg.norm(points, 2, axis=1)
    depth_selected = depth_all[select_index] * 100  # x 100, m to cm.

    # distance to grid center, parallel distance
    distance = distance

    # distance to depth_min, vertical distance
    distance_1 = np.zeros(distance.shape)
    each_group_begin = np.where(index_in_each_group == 0)[0]
    # print(index_in_each_group, each_group_begin, each_group_begin.shape)
    num_valids = len(select_index)
    num_groups = len(each_group_begin)

    for i in range(num_groups):
        begin = each_group_begin[i]
        if (i + 1) < num_groups:
            end = each_group_begin[i + 1]
            distance_1[begin:end] = np.min(depth_selected[begin:end])
        else:
            end = num_valids
            distance_1[begin:end] = np.min(depth_selected[begin:end])

    distance_1 = depth_selected - distance_1
    # calculate_weights
    weight_1 = (1 - distance) ** a
    weight_2 = 1 / (1 + distance_1) ** b
    weight_renew = weight_1 * weight_2
    weight_renew = 1 / (1 + distance + distance_1)
    weight_average = np.float16(weight_renew)  # normalized weights

    group_begin = 0
    cnt = 1
    weight_sum = 0

    for ii in range(num_valids):
        weight_sum = weight_sum + weight_average[ii]
        if cnt < num_groups:
            if (ii + 1) == each_group_begin[cnt]:
                weight_average[group_begin:(ii + 1)] = weight_average[group_begin:(ii + 1)] / weight_sum
                cnt = cnt + 1
                group_begin = ii + 1
                weight_sum = 0
        else:
            end = num_valids
            weight_average[group_begin:end] = weight_average[group_begin:end] / np.sum(weight_average[group_begin:end])
    print(time.time() - start)
    print(weight_average, weight_average.shape)

    return np.float16(weight_average), np.float16(distance_1)


def fun_five(x, a1, a2, a3, a4, a5, a6):
    return a1 * x ** 5 + a2 * x ** 4 + a3 * x ** 3 + a4 * x ** 2 + a5 * x + a6


def fun_six(x, a1, a2, a3, a4, a5, a6, a7):
    return a1 * x ** 6 + a2 * x ** 5 + a3 * x ** 4 + a4 * x ** 3 + a5 * x ** 2 + a6 * x + a7


if __name__ == '__main__':
    first_clip_photo()
    second_choose_colorization_points()
    third_assign_points()
    # fifth_voxel_1_1()
    # sixth_aggregation_1_1()
