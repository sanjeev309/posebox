import os
from PIL import Image


images_dir = "frames"
target_path = "resized_frames"
resize_width = 512
resize_height = 512
for root, folders, files in os.walk(images_dir):
    for file in files:
        img = str(os.path.join(images_dir, file))
        im = Image.open(img)
        size = (resize_width, resize_height)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(target_path + "/" + str(file))
