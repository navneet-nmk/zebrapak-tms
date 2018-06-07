from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cnn_classifier
from keras import losses
from keras import metrics
from keras.callbacks import ModelCheckpoint
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np


def create_data(path, l):
    images = []
    labels = []

    for f in tqdm(os.listdir(path)):
        im = cv2.imread(os.path.join(path, f))
        if im is not None:
            im = cv2.resize(im, (224, 224))
            images.append(im)
            labels.append(l)

    return images, labels


def train(images, labels):

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.4,
        vertical_flip=True,
        shear_range=0.3,
        horizontal_flip=True)

    model, base_model = cnn_classifier.get_cnn_model(weight_init='glorot_normal')
    learning_rate = 0.01
    opt = Adam(lr=learning_rate, decay=learning_rate/1.2)
    model.compile(loss=losses.binary_crossentropy, optimizer=opt, metrics=[metrics.binary_accuracy])
    checkpointer = ModelCheckpoint(filepath="weights.h5", verbose=1)

    model.fit_generator(datagen.flow(images, labels),steps_per_epoch=len(images) / 32,
                        epochs=10, callbacks=[checkpointer])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def predict(path, model, base_model):
    images = []
    encodings = []
    probabilities = []
    for f in tqdm(os.listdir(path)):
        im = cv2.imread(os.path.join(path, f))
        if im is not None:
            im = cv2.resize(im, (224, 224))
            im = np.expand_dims(im, axis=0)
            images.append(im)
    for i in tqdm(images):
        v = model.predict(i)
        encoding = base_model.predict(i)
        encodings.append(encoding)
        probabilities.append(v)

    return encodings, probabilities


def write_to_csv( encodings, probs, path):
    df = pd.DataFrame(
        {
            'encoding':encodings,
            'probabilities':probs
        }
    )
    df.to_csv(path)


if __name__ == '__main__':
    i, l = create_data('data/0', 0)
    i_, l_ = create_data('data/1', 1)
    i.extend(i_)
    l.extend(l_)

    i = np.array(i)
    l = np.array(l)

    train(images=i, labels=l)
    model, base_model = cnn_classifier.get_cnn_model(weight_init='glorot_normal')
    model.load_weights('weights.h5')
    e, p = predict('data/1', model, base_model)
    e_ , p_ = predict('data/0', model, base_model)
    write_to_csv(e_, p_, path='encoding_0.csv')
    write_to_csv(e, p, path='encoding_1.csv')

    e.extend(e_)
    p.extend(p_)

    write_to_csv(e, p, path='encoding.csv')