import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


class TestSetCreator:

    def __init__(self):
        self.images = []
        self.labels = []

    def load_data(self, data_dir, img_Size_X, img_size_Y):
        """
        read all image Files from the Subfolders of a given directory, and
        scales the images.
        Each subfolder represents a class.

        :param data_dir: root directory containing all classes
        :param img_Size_X:  image width
        :param img_size_Y:  image height
        :return: a numpy array of images and depending labels
        """

        directories = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))]

        category = 0
        for d in directories:
            file_names = []
            label_dir = os.path.join(data_dir, d)

            for path, subdirs, files in os.walk(label_dir):
                for name in files:
                    if name.endswith(".jpg") or name.endswith(".png"):
                        os.path.join(path, name)
                        file_names.append(os.path.join(path, name))

            for f in file_names:
                img = cv2.imread(f)
                imresize = cv2.resize(img, (img_Size_X, img_size_Y))

                self.images.append(imresize)
                self.labels.append(category)

            category += 1
        return self.images, self.labels

    def print_img_lbl_array(self):
        print(self.images, self.labels)

    def cross_validate(self, img_arr, label_array, test_size=0.2):
        """
        Split the Image- and Labelarray in train and validation data
        :param img_arr:
        :param label_array:
        :param test_size:
        :return:
        """
        X_train, X_test, y_train, y_test = train_test_split(img_arr, label_array, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test

    def normalize_data(self, img_array, label_array):
        """
        Normalizes the float values of the image data to unsigned 8-bit integer values
        
        :param img_array:
        :param label_array:
        :return:
        """
        image_array = np.array(img_array).astype('float32')
        image_array = image_array / 255
        label_array = np.array(label_array)
        label_array = np_utils.to_categorical(label_array)
        return image_array, label_array

    def shuffel_data(self, img_array, label_array):
        from sklearn.utils import shuffle
        return shuffle(img_array, label_array, random_state=4)
