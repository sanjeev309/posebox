"""
This python script holds configuration for all variables to be used in this project
"""

# Vid 2 Frames
INPUT_VIDEO_PATH = "videos"
OUTPUT_FRAMES_PATH = "frames"
VISUAL = False


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

print("Data visualisation enabled : ", VISUAL)
print("-------------------------------------")