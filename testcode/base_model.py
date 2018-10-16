from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization, Dense, Flatten, Dropout


class BaseModel:

    def createModel(self, img_width, img_heigth, classes, convolution_layers, fully_connected_layer,
                    activation='softmax'):

        model = Sequential()
        model.add(Convolution2D(32, (5, 5), activation='relu', padding='same', input_shape=(img_width, img_heigth, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for layer in convolution_layers:
            self.addConvLayer(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])

        model.add(Flatten())

        for layer in fully_connected_layer:
            self.addFullyConnectedLayers(layer[0], layer[1], layer[2])
        model.add(Dense(classes, activation=activation))
        model.summary()
        return model

    def compile_model(self, model, optimizer='adam', loss_mode='categorical_crossentropy'):
        model.compile(optimizer=optimizer, loss=loss_mode, metrics=['accuracy'])
        return model

    def addConvLayer(self, model, filters, kernelSize_x, kernelSize_y, activation, poolSize, stride):
        model.add(Conv2D(filters, (kernelSize_x, kernelSize_y), padding="same", activation=activation,
                         stride=(stride, stride)))
        model.add(BatchNormalization())
        if poolSize > 0:
            model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

    def addFullyConnectedLayers(self, model, neurons, activation, dropout):
        model.add(Dense(int(neurons), activation=activation))
        model.add(Dropout(dropout))
