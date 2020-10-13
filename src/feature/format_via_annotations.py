import csv
import json
from collections import defaultdict
import os
import cv2

import config


org_annotation_file = config.ANNOTATION_FILE
formatted_annotation_file = config.FORMATTED_ANNOTATION_FILE
header_annotation_file = config.HEADER_ANNOTATION_FILE
images_dir = config.RESIZED_OUTPUT_FRAMES_PATH

"""
Read annotations csv and convert data format and write to new csv
"""
with open(org_annotation_file, 'r') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    align_image_points = defaultdict(lambda: [])
    align_dimensions = defaultdict(lambda: [])
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            image_path = os.path.join(images_dir, row[0])
            image = cv2.imread(image_path)
            width = image.shape[0]
            height = image.shape[1]

            row_5 = json.loads(row[5])
            point = [row_5['cx'], row_5['cy']]
            align_image_points[row[0]] = align_image_points[row[0]] + point
            align_dimensions[row[0]] = [width, height]
            line_count += 1

with open(formatted_annotation_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header_annotation_file)
    for filename, coordinates in align_image_points.items():
        width = align_dimensions[filename][0]
        height = align_dimensions[filename][1]
        writer.writerow([filename, coordinates, width, height])
