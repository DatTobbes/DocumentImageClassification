import os
import numpy as np

from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class PretrainedClassifier:
    def __init__(self, model_path='..\\models\\Transferlearning\\pretrained-Model-150.hdf5'):
        self.model = load_model(model_path)

    def __prediction_to_class(self, arg):
        options = {0: "01_AR",
                   1: "02_ER",
                   2: "03_LN_md",
                   3: "04_LN_sy",
                   4: "05_LB",
                   5: "06_AG"
                   }
        return options.get(arg, "nothing")

    def classify_single_document(self, document_img):
        img = self.__convert_image(document_img)
        prediction = self.model.predict(img)
        class_and_prediction = {"prediction": np.max(prediction), 'class': np.argmax(prediction),
                                'class_name': self.__prediction_to_class(np.argmax(prediction))}
        return class_and_prediction

    def __convert_image(self, image):
        x = image.shape[0]
        y = image.shape[0]
        img = np.array(image).astype('float32')
        reshaped_image = np.reshape(img, [1, x, y, 3])
        return reshaped_image


if __name__ == "__main__":
    img_Size_X = 150
    img_size_Y = 150

    dir = "C:\\tmp\\test2"
    classifier = PretrainedClassifier()

    from utils.create_test_sets import TestSetCreator

    testset = TestSetCreator()
    images, labels = testset.load_data(dir, img_Size_X, img_size_Y)
    x_data, y_data = testset.normalize_data(images, labels)
    x_data, y_data = testset.shuffel_data(x_data, y_data)

    directories = [d for d in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, d))]
    for d in directories:
        file_names = []
        label_dir = os.path.join(dir, d)
    print(directories)

    labela = []
    classa = []
    for index, data in enumerate(x_data):
        predictions = classifier.classify_single_document(data)
        classa.append(directories.index(predictions['class_name']))
        labela.append(np.argmax(y_data[index]))

    print(classa)
    print(labela)

    from utils.confusionmatrix import Confusionmatrix

    cnf = Confusionmatrix(directories)
    cnf_matriy = cnf.compute_confusion_matrix(labela, classa)
    cnf.plot_confusion_matrix(cnf_matriy, False, title='200px')

    print("micro precision=", precision_score(labela, classa, average='micro'))
    print("micro recal=", recall_score(labela, classa, average='micro'))
    print("micro fi=", f1_score(labela, classa, average='micro'))
    print(classification_report(labela, classa, target_names=directories))
