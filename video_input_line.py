from ultralytics import YOLO
import numpy as np

from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator
from supervision.tools.line_counter import LineCounter
from supervision.geometry.dataclasses import Point

from tqdm import tqdm

# import argparse
# import os.path

# Function to read in path and open file
# def is_valid_file(parser, arg):
#     if not os.path.exists(arg):
#         parser.error("%s is not a valid path or file" % arg)
#     else:
#         return open(arg, 'r')  # return an open file handle


# parser = argparse.ArgumentParser(description="Input a vidoe file and count detections from model")
# parser.add_argument("-i", dest="filename", required=True,
#                     help="input video", metavar="FILE",
#                     type=lambda x: is_valid_file(parser, x))

# args = parser.parse_args()

VIDEO_SOURCE = '/home/adrian/Desktop/Capstone/Los Angeles In The Streets - Episode 4.mp4'
TARGET_VIDEO_PATH = '/home/adrian/Desktop/Capstone/testing'

model = YOLO('/home/adrian/Desktop/Capstone/mobile_end/best.pt')
model.fuse()

def main():
   # video_info = sv.VideoInfo.from_video_path(VIDEO_SOURCE)
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    line_zone = sv.LineZone(start = (640, 0), end = (640, 720))
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    box_annotator = sv.BoxAnnotator(sv.ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    generator = sv.get_video_frames_generator(VIDEO_SOURCE)

    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in tqdm(generator, total=video_info.total_frames):
            frame = ...
            sink.write_frame(frame)



if __name__ == "__main__":
    main()