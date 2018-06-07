from keras.layers import Conv2D, MaxPooling2D, Activation, Input, Dense
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2DTranspose
from keras.models import Model


def get_encoder(weights_init):

    input = Input(shape=(224, 224, 3))

    # Encoder
    x = Conv2D(2, 3, padding='same', kernel_initializer=weights_init, activation='relu')(input)
    x = Conv2D(2, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(4, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)
    x = Conv2D(4, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(8, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)
    x = Conv2D(8, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    model = Model(inputs=input, outputs=x)
    return model


def get_decoder(weights_init):

    input = Input(shape=(28, 28, 8))

    # Decoder
    x = Conv2DTranspose(8, 2, strides=(2, 2), padding='same', kernel_initializer=weights_init)(input)
    x = Conv2D(8, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)

    x = Conv2DTranspose(4, 2, strides=(2, 2), padding='same', kernel_initializer=weights_init)(x)
    x = Conv2D(4, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)

    x = Conv2DTranspose(2, 2, strides=(2, 2), padding='same', kernel_initializer=weights_init)(x)
    x = Conv2D(2, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)

    x = Conv2D(3, 3, padding='same', kernel_initializer=weights_init, activation='relu')(x)

    model = Model(inputs=input, outputs=x)
    return model


def get_model(weights_init):

    input= Input(shape=(224, 224, 3))

    encoder= get_encoder(weights_init)
    decoder = get_decoder(weights_init)

    encoded = encoder(input)
    decoded = decoder(encoded)

    model = Model(inputs=input, outputs=decoded)
    return model



