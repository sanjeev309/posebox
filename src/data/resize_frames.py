import os
import config
from PIL import Image

images_dir = config.OUTPUT_FRAMES_PATH
target_path = config.RESIZED_OUTPUT_FRAMES_PATH

resize_width = resize_height = config.RESIZE_DIMEN


def resize_frames():
    for root, folders, files in os.walk(images_dir):
        for file in files:
            img = str(os.path.join(images_dir, file))
            im = Image.open(img)
            size = (resize_width, resize_height)
            im_resized = im.resize(size, Image.ANTIALIAS)
            im_resized.save(target_path + "/" + str(file))

        print("Total Resized Images: ", len(files))


if __name__ == '__main__':
    resize_frames()
