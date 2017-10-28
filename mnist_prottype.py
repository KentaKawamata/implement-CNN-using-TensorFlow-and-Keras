from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import os
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    batch_size = 128
    num_classes = 10
    epochs = 1
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # one-hot表現
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    print("read weight files")
    json_string = open('cnn_model.json', 'r').read()
    model = model_from_json(json_string)
    print("success")

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])
    
    model.load_weights('cnn_model_weights.h5')

    score = model.evaluate(x_test, Y_test, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])

    image = Image.open("testnumber.png").convert('L')
    image = image.resize((28, 28), Image.ANTIALIAS)
    data = np.asarray(image, dtype=float)

    data = data.reshape(1, 28, 28, 1)
    input_shape = (28, 28, 1)
    data = data.astype('float32')
    data /= 255

    classes = model.predict_classes(data, batch_size=32)
    print("予測ラベル: ", classes[0])

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(data.reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()
