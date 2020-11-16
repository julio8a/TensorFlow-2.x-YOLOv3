import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

image_path   = "./IMAGES/kite.jpg"
# video_path   = "./IMAGES/test.mp4"

print("Adding weights")
yolo = Load_Yolo_model()
print("ready to detect")

class Watcher:
    DIRECTORY_TO_WATCH = "/home/pi/PiServer/Julio/machine_learning/watcher/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print (event.src_path)
            if "__" not in event.src_path:
                # *****************************
                # *****************************
                print("Analysing image...")
                detect_image(yolo, event.src_path, event.src_path, input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
                print("Waiting for new image")
                # ******************************
                # ******************************

        # elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            # print("Received modified event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is deleted.
            if "__" not in event.src_path:
                print("Received deleted event - %s." % event.src_path)


if __name__ == '__main__':
    w = Watcher()
    w.run()
