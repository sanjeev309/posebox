import os
import cv2
import time
import config


class Vid2Frames:

    def __init__(self, in_path, out_path, visual):
        self.file_index = 0

        for _, __, files in os.walk(in_path):
            for file in files:
                print("[INFO] Processing: ", file)
                path = os.path.join(in_path, file)
                self.load_video(path)
                self.save_frames(self.video, out_path, visual)

    # def __del__(self):
    #     del self.videos
    #     cv2.destroyAllWindows()

    def load_video(self, path):
        self.video = None

        if os.path.exists(path):
            self.video = cv2.VideoCapture(path)

    def save_frames(self, video, out_path, visual):

        base_filename = "image_"
        frame_count = 0
        while video.isOpened():

            ret, frame = video.read()
            if frame is not None:
                if frame_count % config.SAVE_EVERY == 0:
                    if visual:
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    filename = base_filename + str(self.file_index) + ".jpg"
                    cv2.imwrite(os.path.join(out_path, filename), frame)

                    self.file_index += 1
                frame_count += 1
            else:
                break

        print("[INFO] Total Saved images", self.file_index)
        self.video.release()


if __name__ == "__main__":
    Vid2Frames(config.INPUT_VIDEO_PATH, config.OUTPUT_FRAMES_PATH, config.VISUAL)
