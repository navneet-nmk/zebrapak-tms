from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.applications.mobilenet import MobileNet

def get_encoder(weight_init):
    base_model = MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    dense1 = Dense(64, kernel_initializer=weight_init)(x)

    model = Model(inputs=base_model.input, outputs=dense1)
    return model

def get_cnn_model(weight_init):

    encoder = get_encoder(weight_init=weight_init)
    dense1 = encoder.output

    dense1 = Activation(activation='relu')(dense1)
    dense2 = Dense(1, kernel_initializer=weight_init)(dense1)
    output = Activation(activation='sigmoid')(dense2)

    model = Model(inputs=encoder.input, outputs=output)
    return model, encoder

