import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image
from yolov3.configs import *

Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

print("Adding weights")
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights
print("ready to detect")

class Watcher:
    DIRECTORY_TO_WATCH = "/home/pi/PiServer/Julio/watcher/"

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
                print("Analysing image")
                #detect_image(yolo, event.src_path, "./IMAGES/watchdog.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
                detect_image(yolo, event.src_path, "//IMAGES/watchdog.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
                # ******************************
                # ******************************

        # elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            # print("Received modified event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is deleted.
            print("Received deleted event - %s." % event.src_path)


if __name__ == '__main__':
    w = Watcher()
    w.run()
