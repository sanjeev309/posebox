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

    # Open temp file to write non-empty files and annotations
    with open('new_annotation.csv', 'w') as new_csv_file:
        csv_writer = csv.writer(new_csv_file, delimiter=',')

        for row in csv_reader:
            # Check if empty annotation
            if row[3] == "0":
                filename = row[0]
                try:
                    # Delete that file
                    os.remove(os.path.join(path, filename))
                    print("Deleted: ", filename)
                except:
                    print("Error deleting file: ", filename)
            else:
                csv_writer.writerow(row)
