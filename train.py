import pickle
import os.path
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,Dropout, Conv3D, MaxPooling3D, BatchNormalization
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD

nazwa="400"

DIR = os.path.split(os.path.abspath(__file__))[0]
TDATADIR = os.path.join(DIR, 'trainingdata')
XX = TDATADIR+"/X"+nazwa
yy = TDATADIR+"/y"+nazwa
testX = TDATADIR+"/testX"+nazwa
testy = TDATADIR+"/testy"+nazwa


def loadtrainingdata(pathX, pathy):
    X = pickle.load(open(pathX, "rb"))
    y = pickle.load(open(pathy, "rb"))
    return X, y


def trainmodel(X, y, testX, testy):
    X = X / 255.0
    testX = testX / 255.0

    model = Sequential()

    model.add(Conv2D(32, (4, 4), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    # es = EarlyStopping(min_delta=0.2, patience=7)

    # plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

    history = model.fit(X, y, batch_size=4, epochs=12, validation_data=[testX, testy])

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()
    model.summary()

    return model


def savemodel(model, modelsavepathandname):
    model.save(modelsavepathandname)


X, y = loadtrainingdata(XX, yy)
testX, testy = loadtrainingdata(testX, testy)
model = trainmodel(X, y, testX, testy)
savemodel(model, "model/model400V3")


