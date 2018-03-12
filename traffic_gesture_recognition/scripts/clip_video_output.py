import os
import argparse
from skimage.io import imread, imsave
import glob
import sys

from PIL import ImageFile

def clip_before_detect(image, clip_area):
    rows = image.shape[0]
    cols = image.shape[1]

    # row, col, ch
    clip_img = image[int(rows * clip_area[0]):int(rows * clip_area[1]),
               int(cols * clip_area[2]):int(cols * clip_area[3]), :]
    return clip_img


if __name__ == "__main__":

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    parse = argparse.ArgumentParser()
    parse.add_argument("--imgs_dir", type=str)
    parse.add_argument("--output_dir", type=str, default='clip_video_output')
    parse.add_argument("--clip_area", type=int)

    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    # 1, 2, 4, 6, 7, 8
    camera_clip_area = ((0.5, 1.0, 0.45, 0.7), (0.35, 0.8, 0.45, 0.75),
                        (0.45, 0.9, 0.35, 0.6), (0.44, 0.85, 0.35, 0.65),
                        (0.35, 0.9, 0.15, 0.5), (0.4, 0.78, 0.25, 0.47))

    imgs_dir = flags.imgs_dir
    output_dir = flags.output_dir
    clip_area = camera_clip_area[flags.clip_area]
    path_list = glob.glob(imgs_dir)

    for item in path_list:
        if item.endswith('.png'):
            img = imread(item)
            img = clip_before_detect(img, clip_area)
            tmps = item.split('/')
            tmp_path = os.path.join(tmps[-3], tmps[-2])
            save_path = os.path.join(output_dir, tmp_path)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, tmps[-1])
            imsave(save_path, img)



