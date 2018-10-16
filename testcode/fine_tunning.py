import keras
from keras import callbacks
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class Finetuner:
    """
    This class is the starting point for training a pre-trained model.
    Methods are available to load different architectures of pre-trained models (InceptionV3, VGG16, MobileNet).
    After  a model has been loaded, an output layer must be defined.
    Then the model can then be trained. The TestSetCreator or the
    Keras-ImageDataGenerator API can be used to load the data.

    """

    def __init__(self, img_width, img_height, batchsize):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batchsize
        self.class_weights = 0

    def create_train_test_array(self, Directory, img_size_X, img_size_Y):
        from utils.create_test_sets import TestSetCreator
        data_creator = TestSetCreator()
        images, labels = data_creator.load_data(Directory, img_size_X, img_size_Y)
        self.class_weights = self.get_classweight(labels)
        images, labels = data_creator.normalize_data(images, labels)

        return data_creator.cross_validate(images, labels)

    def set_training_directory(self, Directory):
        self.train_data_dir = Directory

    def set_validation_directory(self, Directory):
        self.validation_data_dir = Directory

    def load_baseModel(self, ModelTyp):
        if 'Inception' in ModelTyp:
            base_model = InceptionV3(weights='imagenet', include_top=False, )
        elif 'MobileNet' in ModelTyp:
            base_model = MobileNet(weights='imagenet', include_top=False,
                                   input_shape=(self.img_width, self.img_height, 3))
        elif 'VGG16' in ModelTyp:
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_width, self.img_height, 3))
        return base_model

    def loadSelfPreTrainedMobileNet(self, modelPath, ModelTyp):

        if 'Inception' in ModelTyp:
            model = load_model(modelPath)
        elif 'MobileNet' in ModelTyp:
            from keras.utils.generic_utils import CustomObjectScope
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(modelPath)
        elif 'VGG16' in ModelTyp:
            model = load_model(modelPath)

        return model

    def __addFullyConnectedLayer(self, model, neurons, classes, activation):

        x = model.output
        # x= Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        # x = Dropout(0.2)(x)
        x = Dense(neurons, kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512, kernel_initializer='uniform', activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(classes, activation=activation)(x)
        newModel = Model(inputs=model.input, outputs=predictions)
        return newModel

    def __freezBaseModel(self, model):

        for layer in model.layers:
            layer.trainable = False

    def __unfreezBaseModel(self, model):

        for layer in model.layers:
            layer.trainable = True

    def ConvLayersToTrain(self, model, unfreezFromLayer):
        for layer in model.layers[:unfreezFromLayer]:
            layer.trainable = False
        for layer in model.layers[unfreezFromLayer:]:
            layer.trainable = True

    def __compileModelForTraining(self, model, optimizer, lossMode, metric):
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=optimizer, loss=lossMode, metrics=[metric])
        return model

    def printModel(self, model):
        model.summary()

    def printLayers(self, model):
        for i, layer in enumerate(model.layers):
            print(i, layer.name)

    def createCallbackList(self, filepath):

        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='auto', period=1)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=250,
                                             verbose=0, mode='auto')

        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                patience=10, min_lr=0.00001)

        tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                                            write_graph=True, write_grads=False, write_images=False,
                                            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        csvLogger = callbacks.CSVLogger('C:/tmp\LearningLogs/log.csv', separator=',', append=False)

        callbacks_list = [checkpoint, reduce_lr, tensorboard, early_stop]

        return callbacks_list

    def createImageDataGenerator(self, horizontalflip, verticalflip, rotationrange, shearrange, channelshift):
        self.datagen = ImageDataGenerator(rescale=1. / 255,
                                          horizontal_flip=horizontalflip,
                                          vertical_flip=verticalflip,
                                          rotation_range=rotationrange,
                                          channel_shift_range=channelshift,
                                          shear_range=shearrange

                                          )

    def createDataGenerator(self, classMode, shuffelData, DataPath):

        generator = self.datagen.flow_from_directory(
            DataPath,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode=classMode,
            shuffle=shuffelData
        )
        samples = int(len(generator.filenames) / self.batch_size)
        return generator, samples

    def trainTheModelWithGenerator(self, model, epochs, trainGenerator, valGenerator, callback_List, stepsPerEpoch,
                                   ValidationSteps):

        model.fit_generator(trainGenerator,
                            steps_per_epoch=stepsPerEpoch,
                            nb_epoch=epochs,
                            validation_data=valGenerator,
                            validation_steps=ValidationSteps,
                            callbacks=callback_List,
                            verbose=2)

        return model

    def get_classweight(self, y, smoth_factor=0.02):
        from collections import Counter
        counter = Counter(y)

        if smoth_factor > 0:
            p = max(counter.values()) * smoth_factor
            for k in counter.keys():
                counter[k] += p

            majority = max(counter.values())

            return {cls: float(majority / count) for cls, count in counter.items()}

    def trainTheModelWithDataArray(self, model, epochs, callback_List, img_size_X, img_size_Y):

        X_train, X_test, y_train, y_test = self.create_train_test_array(self.train_data_dir, img_size_X, img_size_Y)
        print('class_weights: ', self.class_weights)
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  batch_size=self.batch_size,
                  epochs=epochs,
                  callbacks=callback_List,
                  class_weight=self.class_weights,
                  verbose=2)
        return model

    def prepareModelForBaseTraining(self, neurons, classes, modelType, lastLayerActivation, optimizer,
                                    lossMode='categorical_crossentropy'):
        basemodel = self.load_baseModel(modelType)
        model = self.__addFullyConnectedLayer(basemodel, neurons, classes, lastLayerActivation)
        model = self.__compileModelForTraining(model, optimizer, lossMode, 'accuracy')
        self.printModel(model)
        return model

    def prepareModelForFineTunning(self, model, optimizer, lossMode='categorical_crossentropy'):

        self.__unfreezBaseModel(model)
        self.printLayers(model)
        model = self.__compileModelForTraining(model, optimizer, lossMode, 'accuracy')
        self.printModel(model)
        return model


if __name__ == "__main__":
    img_width, img_height = 150, 150
    batchSize = 64
    train_data_dir = 'C:\\tmp\small-RVL\\train'

    finetuner = Finetuner(img_width, img_height, batchSize)

    finetuner.set_training_directory(train_data_dir)

    filepath = 'base-{epoch:02d}-{val_loss:.2f}.hdf5'
    callbacks = finetuner.createCallbackList(filepath)
    optimizer = SGD(lr=0.0005, momentum=0.9)

    model = finetuner.prepareModelForBaseTraining(1024, 6, 'Inception', 'softmax', optimizer)
    model = finetuner.trainTheModelWithDataArray(model, 150, callbacks, img_width, img_height)

    original_data_dir = 'C:\\tmp\\train'
    finetuner.set_training_directory(original_data_dir)
    filepath = 'original-{epoch:02d}-{val_loss:.2f}.hdf5'
    callbacks = finetuner.createCallbackList(filepath)

    model = finetuner.prepareModelForFineTunning(model, optimizer, lossMode='categorical_crossentropy')
    model = finetuner.trainTheModelWithDataArray(model, 150, callbacks, img_width, img_height)
