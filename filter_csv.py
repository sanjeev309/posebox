import os
import csv
import config

path = config.RESIZED_OUTPUT_FRAMES_PATH

"""
Read CSV and delete any image from PATH that is not annotated
"""
with open(config.ANNOTATION_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if row[3] == "0":
            filename = row[0]
            try:
                os.remove(os.path.join(path, filename))
                print("Deleted: ", filename)
            except:
                print("Error deleting file: ", filename)
