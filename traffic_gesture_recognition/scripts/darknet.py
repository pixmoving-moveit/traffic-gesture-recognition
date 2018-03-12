from __future__ import print_function
from ctypes import *
from matplotlib import pyplot as plt
import random
import glob
from skimage.io import imread, imsave
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/j/traffic_classifier/darknet-master/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    # if clip_size is not None:
    #     im = clip_before_detect(im, (0.35, 0.9, 0.15, 0.5))

    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


def clip_and_save_image(result, src_image, save_index, output_file_dir, file_name):

    if result[0] == 'person' and result[1] > 0.7:

        if src_image.shape[0] != 0 and src_image.shape[1] != 0:
            bx = int(np.round(result[2][0] - result[2][2] / 2.0))
            by = int(np.round(result[2][1] - result[2][3] / 2.0))

            bw = int(np.round(result[2][2]))
            bh = int(np.round(result[2][3]))
            bx_l = max((bx, 0))
            by_l = max((by, 0))

            bx_r = bx_l + bw
            by_r = by_l + bh
            bx_r = min((bx_r, src_image.shape[1]))
            by_r = min((by_r, src_image.shape[0]))

            src_image = src_image[by:by_r, bx:bx_r, :]

            file_name = file_name.split('.')[0] + '_' + str(save_index) + '.png'

            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            output_file_path = os.path.join(output_file_dir, file_name)
            print(output_file_path, src_image.shape)
            imsave(output_file_path, src_image)


def detect_and_clip(src_path, save_dir):
    path_list = glob.glob(src_path)

    net = load_net("cfg/yolo.cfg", "yolo.weights", 0)
    meta = load_meta("cfg/coco.data")

    for item in path_list:
        file_paths_split = item.split('/')
        file_name = file_paths_split[-1]
        file_category = file_paths_split[-2]

        # the file path rule : save dir/category/filename: camera_segment_frame.png
        output_file_dir = os.path.join(save_dir, file_category)

        # before detection, clip the image
        detect_result = detect(net, meta, item)

        src_image = imread(item)
        for save_index, result in enumerate(detect_result):
            clip_and_save_image(result, src_image, save_index, output_file_dir, file_name)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--imgs_dir", type=str, default='video_output')
    parse.add_argument("--save_dir", type=str, default='save_dir')
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    imgs_dir = flags.imgs_dir
    save_dir = flags.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    detect_and_clip(imgs_dir, save_dir)



