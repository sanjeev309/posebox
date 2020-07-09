"""
This python script holds configuration for all variables to be used in this project
"""

# Vid 2 Frames
INPUT_VIDEO_PATH = "videos"
OUTPUT_FRAMES_PATH = "frames"
RESIZED_OUTPUT_FRAMES_PATH = "resized_frames"
RESIZE_DIMEN = 512

VISUAL = False

# Frame interval for frame save
SAVE_EVERY = 5


ANNOTATION_FILE = "annotation.csv"

FORMATTED_ANNOTATION_FILE = "annotation_formatted.csv"
HEADER_ANNOTATION_FILE = ['file_name', 'coordinates']

'''
Config validation
'''
import os
import sys

print("-------------------------------------")
if not os.path.exists(INPUT_VIDEO_PATH):
    print("Input folder not found: ", INPUT_VIDEO_PATH)
    sys.exit(0)

if not os.path.exists(OUTPUT_FRAMES_PATH):
    print("Output folder not found: ", OUTPUT_FRAMES_PATH)
    print("Creating folder")
    os.makedirs(OUTPUT_FRAMES_PATH)

if not os.path.exists(RESIZED_OUTPUT_FRAMES_PATH):
    print("Resized Output folder not found: ", RESIZED_OUTPUT_FRAMES_PATH)
    print("Creating folder")
    os.makedirs(RESIZED_OUTPUT_FRAMES_PATH)


print("Data visualisation enabled : ", VISUAL)
print("-------------------------------------")