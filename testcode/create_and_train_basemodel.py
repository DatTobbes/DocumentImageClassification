from keras import callbacks
from keras.optimizers import SGD

from testcode.base_model import BaseModel


def create_callbacks(filename):
    filepath = filename + '-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                            patience=3, min_lr=0.00001)

    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                                        write_graph=True, write_grads=False, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    callbacks_list = [reduce_lr, tensorboard, checkpoint]
    return callbacks_list


def get_classweight(classes, smoth_factor=0.02):
    from collections import Counter
    counter = Counter(classes)

    if smoth_factor > 0:
        p = max(counter.values()) * smoth_factor
        for k in counter.keys():
            counter[k] += p

        majority = max(counter.values())

        return {cls: float(majority / count) for cls, count in counter.items()}


def get_traindata(traind_data_dir, img_height, img_width):
    from utils.create_test_sets import TestSetCreator
    teset_creator = TestSetCreator()
    images, labels = teset_creator.load_data(traind_data_dir, img_height, img_width)
    x_data, y_data = teset_creator.normalize_data(images, labels)
    x_data, y_data = teset_creator.shuffel_data(x_data, y_data)
    X_train, X_test, y_train, y_test = teset_creator.cross_validate(x_data, y_data)
    return X_train, X_test, y_train, y_test


def create_model(img_width, img_height):
    conv_layer1 = [16, 3, 3, 'relu', 2, 1]
    conv_layer2 = [16, 3, 3, 'relu', 2, 1]
    conv_layer3 = [16, 3, 3, 'relu', 0, 1]
    conv_layer4 = [16, 3, 3, 'relu', 2, 1]

    conv_layers = []
    conv_layers.append(conv_layer1)
    conv_layers.append(conv_layer2)
    conv_layers.append(conv_layer3)
    conv_layers.append(conv_layer4)

    fully_connceted_layers = []
    fc1 = [128, 'relu', .2]
    fc2 = [64, 'relu', .1]
    fully_connceted_layers.append(fc1)
    fully_connceted_layers.append(fc2)

    optimizer = SGD(lr=0.001, momentum=0.9)
    model_creater = BaseModel()
    model = model_creater.createModel(img_width, img_height, 1, conv_layers, fully_connceted_layers)
    model = model_creater.compile_model(model, optimizer=optimizer, loss_mode='binary_crossentropy')
    model.summary()
    return model


img_width, img_height = 150, 150
train_data_dir = 'C:\\tmp\\traindata'
batchSize = 16
nb_epoch = 150

X_train, X_test, y_train, y_test = get_traindata(train_data_dir, img_height, img_width)
class_weight = get_classweight(X_test)

model = create_model(img_width, img_height)
model.fit_generator(
    X_train, y_train,
    epochs=nb_epoch,
    validation_data=(X_test, y_test),
    shuffle='batch',
    callbacks=create_callbacks(filename='TestModel'),
    class_weight=class_weight,
    verbose=2)
