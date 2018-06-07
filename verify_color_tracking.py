import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
import cnn_classifier


def load_keras_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d / high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return [h, s, v]


# Capture the input frame from webcam
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()

    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5
    model, base_model = cnn_classifier.get_cnn_model(weight_init='glorot_normal')
    model.load_weights('weights.h5')
    # Iterate until the user presses ESC key
    while True:
        frame = get_frame(cap, scaling_factor)

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
        if c == 27:
            break
        elif c % 256 == 32:
            x_offset = y_offset = 50
            # SPACE pressed
            model_image = cv2.resize(frame, (224, 224))
            model_image = np.expand_dims(model_image, axis=0)
            v = model.predict(model_image)
            print(v)
            #print(base_model.predict(model_image))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Bitwise-AND mask and original image
            if v < 0.5:
                cv2.addWeighted(greenImg, 1, frame, 1, 0, frame)
            else:
                cv2.addWeighted(redImg, 1, frame, 1, 0, frame)
        cv2.imshow('Original image', frame)



    cv2.destroyAllWindows()