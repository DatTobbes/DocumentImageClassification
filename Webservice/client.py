import requests
import json
import shutil
import os
import cv2


class Client:

    def __init__(self):
        from utils.image_converter import ImageConverter
        self.image_converter = ImageConverter()
        self.img_size = 150
        self.threshold = 80

    def __open_image(self, file):
        """
        Calls the converterfunction to convert a document to png
        :param file: 
        :return: img
        """
        return self.image_converter.load_document_image(file)

    def __send_to_classifier(self, img):
        """
        Sends the Document Image to PretrainedClassifier
        :param img: 
        :return: 
        """
        url = "http://127.0.0.1:5000/classify"
        img_data = img.tolist()
        data = {"data": img_data}
        data_json = json.dumps(data)
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, data=data_json, headers=headers)
        return response

    def set_root_directory(self, root_dir):
        self.root_dir = root_dir

    def __get_Classnames(self):
        """
        Get the Classnames from Server
        :return:
        """
        url = "http://127.0.0.1:5000/getClasses"
        response = requests.get(url)
        return json.loads(response.text)

    def create_directories(self, root_dir):
        """
        Checks if there is a Directory for each class
        :return:
        """
        self.set_root_directory(root_dir)
        classnames = self.__get_Classnames()
        for classname in classnames:
            if not os.path.isdir(os.path.join(self.root_dir, classname)):
                os.makedirs(os.path.join(self.root_dir, classname));

    def __move_to_directory(self, file_path, predicted_folder):
        """
        Copy the Document to the predicted Directory
        :param file: Document
        :param predicted_folder: predicted Class
        :return:
        """
        copy_from = file_path
        copy__to = os.path.join(self.root_dir, predicted_folder, os.path.basename(file_path))
        print(copy_from)
        print(copy__to)
        shutil.move(copy_from, copy__to)

    def classify_document_and_move(self, file_path):
        img = self.__open_image(file_path)
        response = self.__send_to_classifier(
            cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA))
        json_data = json.loads(response.text)
        self.__move_to_directory(file_path, json_data[0]['classname'])
        return img, json_data[0]

    def classify_document(self, file_path):
        img = self.__open_image(file_path)
        response = self.__send_to_classifier(
            cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA))
        json_data = json.loads(response.text)
        return img, json_data[0]


if __name__ == "__main__":
    client = Client('C:\\tmp\\test')
    dir = 'C:\\tmp\\test'
    file = "test.pdf"
    client.create_directories()
    client.classify_document(os.path.join(dir, file))
