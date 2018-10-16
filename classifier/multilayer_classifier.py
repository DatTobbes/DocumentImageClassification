import os
import math
import numpy as np
from keras.models import load_model


class MultilayerClassifier:

    def __init__(self):
        self.intern_extern_model = load_model('..\\models\\Intern-Extern-Model\\150px\\binary.hdf5', compile=True)
        self.intern_model = load_model('..\\models\\Intern-Extern-Model\\150px\\intern.hdf5', compile=True)
        self.extern_model = load_model('..\\models\\Intern-Extern-Model\\150px\\extern.hdf5', compile=True)

    def __prediction_to_class(self, arg):
        options = {0: "02_ER",
                   1: "04_LN_sy",
                   2: "01_AR",
                   3: "03_LN_md",
                   4: "05_LB",
                   5: "06_AG"
                   }
        return options.get(arg, "nothing")

    def __calc_extern_prediction(self, prediction):
        return math.fabs((prediction - 0.5) / 0.5) * 100

    def __clssify_intern_extern(self, document_img):

        prediction = self.intern_extern_model.predict(document_img)
        class_and_prediction = {"prediction": prediction, 'class': int(round(prediction.item(0)))}
        return class_and_prediction

    def __classify_intern_document(self, document_img):
        prediction = self.intern_model.predict(document_img)
        class_and_prediction = {"prediction": np.max(prediction), 'class': np.argmax(prediction) + 2,
                                'class_name': self.__prediction_to_class(np.argmax(prediction) + 2)}
        return class_and_prediction

    def get_label_dir(self, folders):

        directories = [d for d in os.listdir(folders)
                       if os.path.isdir(os.path.join(folders, d))]
        print(directories)
        return directories

    def __classify_extern_document(self, document_img):

        prediction = self.extern_model.predict(document_img)
        class_and_prediction = {"prediction": self.__calc_extern_prediction(prediction),
                                'class': int(round(prediction.item(0))),
                                'class_name': self.__prediction_to_class(int(round(prediction.item(0))))}
        return class_and_prediction

    def classify_single_document(self, document_img):

        img = self.__convert_image(document_img)
        intern_extern_prediction = self.__clssify_intern_extern(img)

        if intern_extern_prediction['class'] == 1:
            return self.__classify_intern_document(img)

        elif intern_extern_prediction['class'] == 0:
            return self.__classify_extern_document(img)

    def __convert_image(self, image):
        x = image.shape[0]
        y = image.shape[0]
        img = np.array(image).astype('float32')
        reshaped_image = np.reshape(img, [1, x, y, 3])
        return reshaped_image


if __name__ == "__main__":

    classificator = MultilayerClassifier()
    data_dir = "C:\\tmp\\test2"
    directories = classificator.get_label_dir(data_dir)
    from utils.evaluate_model import Evaluator

    evaluator = Evaluator()
    classNames = evaluator.get_class_names(data_dir)
    test_X, test_Y = evaluator.get_test_data(data_dir, 150, 150)

    true_classes = evaluator.labels_to_classes(test_Y)
    predicted_classes = []

    for index, data in enumerate(test_X):
        predictions = classificator.classify_single_document(data)
        predicted_classes.append(directories.index(predictions['class_name']))

    print("macro recal=", evaluator.recall(true_classes, predicted_classes, 'weighted'))
    print("micro recal=", evaluator.recall(true_classes, predicted_classes, 'micro'))
    print("macro precision=", evaluator.presision_score(true_classes, predicted_classes, 'weighted'))
    print("micro precision=", evaluator.presision_score(true_classes, predicted_classes, 'micro'))
    print("macro f1=", evaluator.f1_score(true_classes, predicted_classes, 'weighted'))
    print("micro f1=", evaluator.f1_score(true_classes, predicted_classes, 'micro'))
    print('report: ', evaluator.classification_report(true_classes, predicted_classes, classNames))
    print('accurary:', evaluator.accuracy_score(true_classes, predicted_classes))

    from utils.confusionmatrix import Confusionmatrix

    cnf = Confusionmatrix(directories)
    cnf_matriy = cnf.compute_confusion_matrix(true_classes, predicted_classes)
    cnf.plot_confusion_matrix(cnf_matriy, False, title='150px')
