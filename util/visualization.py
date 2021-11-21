# coding=gbk
import glob
import cv2
import numpy as np
import os
from msvcrt import getch
import time
import win32api
import win32con

# def combine_img(id):
#     path1 = "./result/pointcloud/"
#     path2 = "./result/NPCR-MP/"
#     path3 = "./result/NPG/"
#     path4 = "./result/LF-FCN/"
#     path5 = "./result/pix2pixHD/"
#     path6 = "./result/MPR-GAN/"
#
#     debug_images = []
#     img1 = cv2.imread(path1 + str(id) + '.png')
#     img2 = cv2.imread(path2 + str(id) + '.png')
#     img3 = cv2.imread(path3 + str(id) + '.png')
#     img4 = cv2.imread(path4 + str(id) + '.png')
#     img5 = cv2.imread(path5 + str(id) + '.jpg')
#     img6 = cv2.imread(path6 + str(id) + '.jpg')
#
#     a = 1.6
#     img1 = cv2.resize(img1, (int(img1.shape[1] / a), int(img1.shape[0] / a)))
#     img2 = cv2.resize(img2, (int(img2.shape[1] / a), int(img2.shape[0] / a)))
#     img3 = cv2.resize(img3, (int(img3.shape[1] / a), int(img3.shape[0] / a)))
#     img4 = cv2.resize(img4, (int(img4.shape[1] / a), int(img4.shape[0] / a)))
#     img5 = cv2.resize(img5, (int(img5.shape[1] / a), int(img5.shape[0] / a)))
#     img6 = cv2.resize(img6, (int(img6.shape[1] / a), int(img6.shape[0] / a)))
#
#     debug_images.append(img1)
#     debug_images.append(img2)
#     debug_images.append(img3)
#     debug_images.append(img4)
#     debug_images.append(img5)
#     debug_images.append(img6)
#
#     return debug_images
#
#
# def show_in_one(images, show_size=(120 * 3 + 3 * 4, 480 * 2 + 2 * 3), blank_size1=3, blank_size2=2):
#     small_h, small_w = images[0].shape[:2]
#     column = int(show_size[1] / (small_w + blank_size2))
#     row = int(show_size[0] / (small_h + blank_size1))
#
#     shape = [show_size[0], show_size[1]]
#     for i in range(2, len(images[0].shape)):
#         shape.append(images[0].shape[i])
#     merge_img = np.zeros(tuple(shape), images[0].dtype)
#     max_count = len(images)
#     count = 0
#     for i in range(row):
#         if count >= max_count:
#             break
#         for j in range(column):
#             if count < max_count:
#                 im = images[count]
#                 t_h_start = i * small_h + (i + 1) * blank_size1
#                 t_w_start = j * small_w + (j + 1) * blank_size2
#                 t_h_end = t_h_start + im.shape[0]
#                 t_w_end = t_w_start + im.shape[1]
#                 merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
#                 count = count + 1
#             else:
#                 break
#     if count < max_count:
#         print("图片总数为： %s" % (max_count - count))
#     return merge_img
#
#
# if __name__ == '__main__':
#
#     i = 0
#     images = combine_img(i)
#     merge_img = show_in_one(images)
#     cv2.imshow('xxx', merge_img)
#
#     while True:
#         k = cv2.waitKey(0)
#         print(k)
#         time.sleep(0.15)
#         win32api.keybd_event(0x53, 0, 0, 0)
#         if k == ord('s'):
#             i = i + 1
#             if i == 65:
#                 break
#             images = combine_img(i)
#             merge_img = show_in_one(images)
#             cv2.imshow('xxx', merge_img)
#         elif k == ord('a'):
#             break
#
#     cv2.destroyAllWindows()


def combine_img(id):
    path1 = "./result/pc_scene2/"
    path2 = "./result/MPR-GAN2/"

    debug_images = []
    img1 = cv2.imread(path1 + str(id) + '.png')
    img2 = cv2.imread(path2 + str(id) + '.jpg')

    a = 1.0
    img1 = cv2.resize(img1, (int(img1.shape[1] / a), int(img1.shape[0] / a)))
    img2 = cv2.resize(img2, (int(img2.shape[1] / a), int(img2.shape[0] / a)))

    debug_images.append(img1)
    debug_images.append(img2)

    return debug_images


def show_in_one(images, show_size=(192 * 2 + 2 * 3, 768 * 1 + 60 * 2), blank_size1=2, blank_size2=50):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size2))
    row = int(show_size[0] / (small_h + blank_size1))

    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])
    merge_img = np.zeros(tuple(shape), images[0].dtype)
    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * small_h + (i + 1) * blank_size1
                t_w_start = j * small_w + (j + 1) * blank_size2
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("图片总数为： %s" % (max_count - count))
    return merge_img


if __name__ == '__main__':

    i = 1
    images = combine_img(i)
    merge_img = show_in_one(images)
    cv2.imshow('xxx', merge_img)

    while True:
        k = cv2.waitKey(0)
        print(k)
        time.sleep(0.15)
        win32api.keybd_event(0x53, 0, 0, 0)
        if k == ord('s'):
            i = i + 1
            if i == 200:
                break
            images = combine_img(i)
            merge_img = show_in_one(images)
            cv2.imshow('xxx', merge_img)
        elif k == ord('a'):
            break

    cv2.destroyAllWindows()


