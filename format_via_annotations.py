import csv
import json
from collections import defaultdict
import config


org_annotation_file = config.ANNOTATION_FILE
formatted_annotation_file = config.FORMATTED_ANNOTATION_FILE
header_annotation_file = config.HEADER_ANNOTATION_FILE

with open(org_annotation_file, 'r') as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    align_image_points = defaultdict(lambda: [])
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            row_5 = json.loads(row[5])
            point = [row_5['cx'], row_5['cy']]
            align_image_points[row[0]] = align_image_points[row[0]] + point
            line_count += 1

with open(formatted_annotation_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header_annotation_file)
    for key, value in align_image_points.items():
        writer.writerow([key, value])
