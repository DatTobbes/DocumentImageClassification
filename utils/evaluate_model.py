import os
import numpy as np

from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, \
    accuracy_score
from utils.confusionmatrix import Confusionmatrix
from utils.create_test_sets import TestSetCreator


class Evaluator:
    """
    This Class helps to evaluate a CNN Model

    There a methods to compute diffrent metrics like
    accuracy, precission, recal and f1. Its also possible
    to draw a confusion matrix with this class.

    """

    def __init__(self):
        self.test_set_creator = TestSetCreator()

    def get_test_data(self, directory, img_size_X, img_size_Y):
        images, labels = self.test_set_creator.load_data(directory, img_size_X, img_size_Y)
        x_data, y_data = self.test_set_creator.normalize_data(images, labels)
        return self.test_set_creator.shuffel_data(x_data, y_data)

    def set_model(self, model):
        self.model = model

    def get_prediction(self, test_data):
        return self.model.predict(test_data)

    def get_predicted_classes(self, predictions):
        return np.around(predictions)

    def evaluate_model(self, test_data, test_labels):
        """
        :param test_data: a nupmy array of images to test
        :param test_labels: a numpy array off image labels
        :return: loss and accuray
        """
        values = {}
        test_labels = self.labels_to_classes(test_labels)
        metrics = self.model.evaluate(test_data, test_labels, verbose=1)
        values['loss'] = metrics[0]
        values['accuracy'] = metrics[1]
        return values

    def get_class_names(self, directory):
        """
        read the subfoldernames in a directory as classnames
        :param directory:
        :return:a array of classnames
        """
        return next(os.walk(directory))[1]

    def labels_to_classes(self, hot_encoded_label_array):
        """
        Converts a binary Labelarray to a Integernumber
        :param hot_encoded_label_array:
        :return:
        """
        label_array = []
        for label in hot_encoded_label_array:
            label_array.append(np.argmax(label))
        return np.asarray(label_array)

    def predictions_to_classes(self, prediction_array, class_names):
        """
        :param prediction_array: 
        :param class_names: 
        :return: 
        """
        predicted_class_array = []
        if len(class_names) > 2:
            for prediction in prediction_array:
                predicted_class_array.append(np.argmax(prediction))
        elif len(class_names) == 2:
            for prediction in prediction_array:
                predicted_class_array.append(int(round(prediction[0])))
        return predicted_class_array

    def make_confudsion_matrix(self, class_names, predictions, labels, titel):
        """
        compute a confusionmatrix from predictions made with the model
        :param class_names: a array of classnames
        :param predictions: a array of predicted classes as integers
        :param labels: integer array of labels
        :param titel: title to print
        :return:
        """
        matrix_maker = Confusionmatrix(class_names)
        predictions = self.predictions_to_classes(predictions, class_names)
        labels = self.labels_to_classes(labels)
        cnf_matrix = matrix_maker.compute_confusion_matrix(labels, predictions)
        matrix_maker.plot_confusion_matrix(cnf_matrix, False, titel)

    def presision_score(self, labels, predictions, average='None'):
        return precision_score(labels, predictions, average=average)

    def recall(self, labels, predictions, average='None'):
        return recall_score(labels, predictions, average=average)

    def f1_score(self, labels, predictions, average='None'):
        return f1_score(labels, predictions, average=average)

    def cohen_kappa(self, labels, predictions):
        return cohen_kappa_score(labels, predictions)

    def classification_report(self, labels, predictions, classnames):
        return (classification_report(labels, predictions, target_names=classnames))

    def accuracy_score(self, labels, predictions):
        return accuracy_score(labels, predictions)

    def get_classweight(self, classes, smoth_factor=0.02):
        """
        compute the class weights for test data
        counts file in all classes and calculate the ratio

        :param classes: interger array of labels
        :param smoth_factor:
        :return:
        """

        from collections import Counter
        counter = Counter(classes)

        if smoth_factor > 0:
            p = max(counter.values()) * smoth_factor
            for k in counter.keys():
                counter[k] += p

            majority = max(counter.values())

            return {cls: float(majority / count) for cls, count in counter.items()}


if __name__ == "__main__":

    model = load_model('..\Models\\Intern-Extern-Model\\150px\\extern.hdf5')
    data_dir = "C:\\tmp\\test"
    evaluator = Evaluator()
    evaluator.set_model(model)
    classNames = evaluator.get_class_names(data_dir)
    test_X, test_Y = evaluator.get_test_data(data_dir, 150, 150)

    predictions = evaluator.get_prediction(test_X)
    evaluator.make_confudsion_matrix(classNames, predictions, test_Y, '150px')
    labels = evaluator.labels_to_classes(test_Y)

    for index, label in enumerate(labels):
        if label != np.argmax(predictions[index]):
            print(label, predictions[index])

    class_weight = evaluator.get_classweight(labels)
    print('class_weight:', class_weight)

    predictions = evaluator.predictions_to_classes(predictions, classNames)
    print("macro recal=", evaluator.recall(labels, predictions, 'weighted'))
    print("micro recal=", evaluator.recall(labels, predictions, 'micro'))
    print("macro precision=", evaluator.presision_score(labels, predictions, 'weighted'))
    print("micro precision=", evaluator.presision_score(labels, predictions, 'micro'))
    print("macro f1=", evaluator.f1_score(labels, predictions, 'weighted'))
    print("micro f1=", evaluator.f1_score(labels, predictions, 'micro'))
    print('report: ', evaluator.classification_report(labels, predictions, classNames))

    print('accurary:', evaluator.accuracy_score(labels, predictions))
