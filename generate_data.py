import os
import cv2
import csv

import numpy as np

num_output = 8
input_shape = (512, 512, 3)

batch_size = 10

IMAGES_FOLDER = 'resized_frames'
ANNOTATION_FILE = 'annotation_formatted.csv'
OUTPUT = 'output'

### Initialise empty numpy arrays

data = np.empty((0,512,512,3), dtype=np.int8)
target = np.empty((0,8), dtype=np.float)

### Read annotation file, fetch image, normalise image and array, compose data and target arrays

with open(ANNOTATION_FILE,'r') as csv_file:

    reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in reader:

        # print(row)

        if line_count == 0:
            line_count += 1
        else:
            image_path = os.path.join(IMAGES_FOLDER, row[0])
            image = cv2.imread(image_path)/ 255
            image = np.expand_dims(image, axis=0)

            points = row[1]
            dimen = (float)(row[2])

            p = points.strip('][').split(', ')


            p = np.array(p, dtype=np.int)
            p = np.divide(p, dimen)
            p = np.expand_dims(p, axis=0)

            if image is not None:
                data = np.vstack((data, image))
                target = np.vstack((target, p))

            line_count += 1

### Shuffle data and target synchronously

num_samples = data.shape[0]
arr = np.arange(num_samples)
np.random.shuffle(arr)
print("num_samples", num_samples)
data = data[arr]
target = target[arr]

print(data.shape)
print(target.shape)

np.save(os.path.join(OUTPUT,'data.npy'), data)
np.save(os.path.join(OUTPUT,'target.npy'), target)
