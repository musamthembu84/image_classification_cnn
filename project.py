from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def print_labels():
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print('-----------------------------')



index = 0
x_train[index]

def create_classification():
    

    print('The image label is:', y_train[index])
    classification = ['airplane', 'automobile', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print('The image class is: ', classification[y_train[index][0]])
    
    print('-----------------------------')

   


def convolutional_layer():

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    print(y_train_one_hot)
    print('One hot label is :', y_train_one_hot[0])

    x_trains = x_train / 225
    x_tests = x_test / 225


    #Building convolution neural network model 
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(1000, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(500, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(250, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist =  model.fit(x_trains, y_train_one_hot,
                    batch_size=256, epochs=10, validation_split=0.2)


    model.evaluate(x_tests, y_test_one_hot)[1]
    
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    #Visualize the models loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


 if __name__ == "__main__" :
    print_labels()
    create_classification()
    convolutional_layer()
    