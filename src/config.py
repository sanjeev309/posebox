"""
This python script holds configuration for all variables to be used in this project
"""

# Vid 2 Frames
INPUT_VIDEO_PATH = "../data/raw"
OUTPUT_FRAMES_PATH = "../data/interim"
RESIZED_OUTPUT_FRAMES_PATH = "../data/processed"
EXTERNAL = "../data/external"
RESIZE_DIMEN = 512

VISUAL = False

# Frame interval for frame save
SAVE_EVERY = 5


ANNOTATION_FILE = "../data/interim/annotation.csv"

FORMATTED_ANNOTATION_FILE = "../../data/processed/formatted_annotation.csv"
CLEAN_ANNOTATION_FILE = "../../data/processed/clean_annotation.csv"
HEADER_ANNOTATION_FILE = ['file_name', 'coordinates', 'width', 'height']


OUTPUT = '../models'

TEST_OUTPUT = "../eval"