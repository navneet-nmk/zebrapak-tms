import cv2
import numpy as np
from keras.models import model_from_json
import keras.backend as K
from PIL import Image
import cnn_classifier
import tensorflow as tf


class VideoCamera(object):
    def __init__(self, model, graph):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.model = model
        self.graph = graph

        print(self.model.summary())
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream

        scaling_factor = 0.5

        while True:
            frame = cv2.resize(image, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)

            # Convert the HSV colorspace
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #

            # Define 'blue' range in HSV colorspace
            lower = np.array([45, 100, 100])
            upper = np.array([85, 255, 255])

            # Threshold the HSV image to get only blue color
            mask = cv2.inRange(hsv, lower, upper)

            # Creating a red mask
            redImg = np.zeros(frame.shape, frame.dtype)
            redImg[:, :] = (0, 0, 255)
            greenImg = np.zeros(frame.shape, frame.dtype)
            greenImg[:, :] = (0, 255, 0)
            # Check if the user pressed ESC key
            c = cv2.waitKey(5)
            x_offset = y_offset = 50
            # SPACE pressed
            model_image = cv2.resize(frame, (224, 224))
            model_image = np.expand_dims(model_image, axis=0)
            v = -1
            with self.graph.as_default():
                v = self.model.predict(model_image)
                #K.clear_session()
                print(v)
            # Bitwise-AND mask and original image
            if v < 0.5:
                cv2.addWeighted(greenImg, 1, frame, 1, 0, frame)
            else:
                cv2.addWeighted(redImg, 1, frame, 1, 0, frame)

            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()