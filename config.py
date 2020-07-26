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
HEADER_ANNOTATION_FILE = ['file_name','coordinates', 'width', 'height']


OUTPUT = '/home/sanjeev309/Projects/posebox/output'

TEST_OUTPUT = "/home/sanjeev309/Projects/posebox/smoke_test"