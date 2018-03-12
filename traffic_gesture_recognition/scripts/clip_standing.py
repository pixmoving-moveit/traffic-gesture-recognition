from __future__ import print_function
import clip_video
import argparse
import sys
import os
import datetime
from skimage.io import imread, imsave
from moviepy.editor import *
from PIL import ImageFile

class state(object):
    def __init__(self):
        self.id = 0
        self.time = ''
        self.category = 0

        self.start_time = ''
        self.end_time = ''

def clip_before_detect(image, clip_area):
    rows = image.shape[0]
    cols = image.shape[1]

    # row, col, ch
    clip_img = image[int(rows * clip_area[0]):int(rows * clip_area[1]),
               int(cols * clip_area[2]):int(cols * clip_area[3]), :]
    return clip_img

def parse_time(offset, t):
    tt = datetime.datetime.strptime(t, '%H:%M:%S,%f')
    tt += datetime.timedelta(seconds=offset)
    return tt.strftime('%H:%M:%S.%f')


def time_process(clip_time, offset):
    clip_time = clip_time.split('-->')

    # get the clip start and end time
    start_time = clip_time[0]
    end_time = clip_time[1]
    start_time = start_time.strip()
    end_time = end_time.strip()

    # replace the , with . and add a small offset (about 300')
    start_time_ = parse_time(offset, start_time)
    end_time_ = parse_time(offset, end_time)
    return start_time_, end_time_


def parse_video(parse_result, video_path, output_path, clip_area, offset):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    pre_id = 0
    pre_category = 0
    pre_time = ''

    for i in range(0, len(parse_result) - 1, 3):

        id = parse_result[i]
        clip_time = parse_result[i + 1]
        category = parse_result[i + 2]

        if category != 4:
            if pre_category != 0:
                print(pre_time)
                pre_start_time, pre_end_time = time_process(pre_time, offset)
                start_time, end_time = time_process(clip_time, offset)

                print('Clipping frame from : ', pre_end_time, start_time)

                clip = VideoFileClip(video_path, audio=False).subclip(pre_end_time, start_time)

                if video_path.endswith('.webm'):
                    clip.fps = 30
                frame_id = 1
                for frame in clip.iter_frames():
                    input_file_name = video_path.split('/')[-2]

                    # the save file name: video file + duration id + frame_id
                    save_file_name = input_file_name + '_' + id + '_' + str(frame_id) + '.png'

                    tmp_path = os.path.join(output_path, category)
                    # if there is no category dir , mk it
                    if not os.path.exists(tmp_path):
                        os.mkdir(tmp_path)

                    save_path = os.path.join(tmp_path, save_file_name)

                    clip_img = clip_before_detect(frame, clip_area)
                    # print('save file', save_path)
                    imsave(save_path, clip_img)
                    frame_id = frame_id + 1

                    pre_id = id
                    pre_category = category
                    pre_time = clip_time
            else:
                pre_id = id
                pre_category = category
                pre_time = clip_time
        else:
            pre_id = 0
            pre_category = 0
            pre_time = ''


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parse = argparse.ArgumentParser()
    parse.add_argument("--video_path", type=str, help="the video to be clip")
    parse.add_argument("--save_dir", type=str, default='standing_output/')
    parse.add_argument("--offset", type=float, default=0.0)
    parse.add_argument('--srt_file', type=str, default='video_data/record.srt')
    parse.add_argument("--clip_area", type=int)
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    save_dir = flags.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 1, 2, 4, 6, 7, 8
    camera_clip_area = ((0.5, 1.0, 0.45, 0.7), (0.35, 0.8, 0.45, 0.75),
                        (0.45, 0.9, 0.35, 0.6), (0.44, 0.85, 0.35, 0.65),
                        (0.35, 0.9, 0.15, 0.5), (0.4, 0.78, 0.25, 0.47))

    srt_result = clip_video.parse_srt_file(flags.srt_file)

    parse_video(srt_result, flags.video_path, save_dir , camera_clip_area[flags.clip_area], flags.offset)
