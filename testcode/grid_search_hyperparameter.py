import numpy as np
from keras import callbacks
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

"""
This script is used to create models and determine hyperparameters 
such as batch size, training epochs, and loss functions by gridsearch.

For this purpose, the Basemodel class is used to create models and the 
TestSetCreator-class is used to load the data
"""


def create_model(optimizer='rmsprop', img_width=150, img_height=150):
    convlayer1 = [16, 3, 3, 'relu', 2, 1]
    convlayer2 = [16, 3, 3, 'relu', 2, 1]
    convlayer3 = [16, 3, 3, 'relu', 0, 1]
    convlayer4 = [16, 3, 3, 'relu', 2, 1]

    convolutional_layers = []
    convolutional_layers.append(convlayer1)
    convolutional_layers.append(convlayer2)
    convolutional_layers.append(convlayer3)
    convolutional_layers.append(convlayer4)

    fully_connceted_layers = []
    fc1 = [128, 'relu', .2]
    fc2 = [64, 'relu', .1]

    fully_connceted_layers.append(fc1)
    fully_connceted_layers.append(fc2)

    from testcode.base_model import BaseModel
    model_creator = BaseModel()
    model = model_creator.createModel(img_width, img_height, 1, convolutional_layers, fully_connceted_layers)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_dataset(data_dir, img_width, img_height):
    from utils.create_test_sets import TestSetCreator
    testset_creator = TestSetCreator()
    images, labels = testset_creator.load_data(data_dir, img_width, img_height)
    images, labels = testset_creator.normalize_data(images, labels)
    X_train, X_test, y_train, y_test = testset_creator.cross_validate(images, labels, 0)
    return X_train, y_train


def create_callbacks():
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.20,
                                            patience=5, min_lr=0.00001)

    tensorboard = callbacks.TensorBoard(log_dir='./logs',
                                        histogram_freq=0,
                                        batch_size=16,
                                        embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None)

    return [reduce_lr, tensorboard]


X_train, y_train = create_dataset('C:\\tmp\\test', 150, 150)
model = KerasClassifier(build_fn=create_model, verbose=0)

# define Hyperparamter to Test in Gridsearch
optimizers = [SGD(lr=0.001, momentum=0.9), 'adam']
epochs = [100, 200, 300]
batches = [8, 16, 32, 64]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
