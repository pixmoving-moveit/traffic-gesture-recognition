from __future__ import print_function

from skimage.io import imsave
import os
import datetime
import argparse
import sys
from moviepy.editor import *


def parse_time(offset, t):
    tt = datetime.datetime.strptime(t, '%H:%M:%S,%f')
    tt += datetime.timedelta(seconds=offset)
    return tt.strftime('%H:%M:%S.%f')


def parse_video(video_path, srt_path, output_path, offset=0):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    parse_result = []
    with open(srt_path, 'r') as f:
        lines = f.readlines()

        for item in lines:
            if item.startswith('\r\n'):
                continue
            else:
                item = item.strip()
                parse_result.append(item)
        f.close()

    print(parse_result)

    for i in range(0, len(parse_result) - 1, 3):
        id = parse_result[i]
        clip_time = parse_result[i + 1]
        category = parse_result[i + 2]

        clip_time = clip_time.split('-->')

        # get the clip start and end time
        start_time = clip_time[0]
        end_time = clip_time[1]
        start_time = start_time.strip()
        end_time = end_time.strip()

        # replace the , with . and add a small offset (about 300')
        start_time_ = parse_time(offset-0.3, start_time)
        end_time_ = parse_time(offset, end_time)
        print(start_time, end_time, start_time_, end_time_)
        
        clip = VideoFileClip(video_path, audio=False).subclip(start_time_, end_time_)

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
            imsave(save_path, frame)
            frame_id = frame_id + 1


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--video_path", type=str, help="the video to be clip")
    parse.add_argument("--save_dir", type=str, default='video_output/')
    parse.add_argument("--offset", type=float, default=0.0)
    parse.add_argument('--srt_file', type=str, default='video_data/record.srt')
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    parse_video(flags.video_path, flags.srt_file, flags.save_dir, offset=flags.offset)



    


