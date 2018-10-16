from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
from utils.Opener import Opener
import cv2
from PyQt5.QtGui import *
from webservice.client import Client
from utils.image_converter import ImageConverter


class UiPicturesViewWindow(object):

    def __init__(self):
        super().__init__()
        self.filePathArray = []
        self.counter = 0

    def setupUi(self, PicturesViewWindow):
        PicturesViewWindow.setObjectName("Document Classification")
        PicturesViewWindow.resize(650, 1000)
        self.grid_layout = QtWidgets.QGridLayout(PicturesViewWindow)
        self.grid_layout.setObjectName("gridLayout")
        self.grid_layout_2 = QtWidgets.QGridLayout()
        self.grid_layout_2.setObjectName("gridLayout_2")

        self.graphics_view = QLabel(PicturesViewWindow)
        self.graphics_view.setObjectName("graphicsView")
        self.grid_layout_2.addWidget(self.graphics_view, 0, 2, 1, 1)

        self.prediction_label = QLabel(PicturesViewWindow)
        self.prediction_label.setObjectName("predictionLabel")
        self.prediction_label.setFixedHeight(15)
        self.grid_layout_2.addWidget(self.prediction_label, 1, 2, 1, 1)

        self.next_picturs_btn = QtWidgets.QPushButton(PicturesViewWindow)
        self.next_picturs_btn.setObjectName("nextPicturs_btn")
        self.grid_layout_2.addWidget(self.next_picturs_btn, 2, 2, 1, 1)

        self.grid_layout.addLayout(self.grid_layout_2, 0, 0, 1, 1)
        self.horizontal_layout_2 = QtWidgets.QHBoxLayout()
        self.horizontal_layout_2.setObjectName("horizontalLayout_2")

        self.open_pictures_btn = QtWidgets.QPushButton(PicturesViewWindow)
        self.open_pictures_btn.setObjectName("openPictures_btn")
        self.horizontal_layout_2.addWidget(self.open_pictures_btn)

        self.multi_classification_btn = QtWidgets.QPushButton(PicturesViewWindow)
        self.multi_classification_btn.setObjectName("multi_classification_btn")
        self.horizontal_layout_2.addWidget(self.multi_classification_btn)

        self.grid_layout.addLayout(self.horizontal_layout_2, 1, 0, 1, 1)
        self.graphics_view.raise_()
        self.retranslateUi(PicturesViewWindow)
        QtCore.QMetaObject.connectSlotsByName(PicturesViewWindow)

    def retranslateUi(self, PicturesViewWindow):
        self.__init_side_Objects()
        _translate = QtCore.QCoreApplication.translate
        PicturesViewWindow.setWindowTitle(_translate("PicturesViewWindow", "Classify Pictues"))
        self.next_picturs_btn.setText(_translate("PicturesViewWindow", "Classify next Document"))
        self.next_picturs_btn.clicked.connect(self.single_file)
        self.open_pictures_btn.setText(_translate("PicturesViewWindow", "Open Document Folder"))
        self.open_pictures_btn.clicked.connect(self.load_images)
        self.multi_classification_btn.setText(_translate("PicturesViewWindow", "Multiple Classification"))
        self.multi_classification_btn.clicked.connect(self.multi_file)

    def __init_side_Objects(self):
        self.opener = Opener()
        self.converter = ImageConverter()
        self.client = Client()
        self.client.create_directories(self.opener.get_root_dir())

    def load_images(self):
        self.filePathArray = self.opener.get_documents_from_directory()
        from sklearn.utils import shuffle
        self.filePathArray = shuffle(self.filePathArray, random_state=4)
        self.single_file()
        print()

    def single_file(self):
        try:
            img, response = self.client.classify_document_and_move(self.filePathArray[self.counter])
            pixmap = self.__convert_img_to_qt(img)
            print(self.filePathArray[self.counter])
            width = self.graphics_view.width()
            height = self.graphics_view.height()
            scaled_pic = pixmap.scaledToWidth(width)
            scaled_pic = scaled_pic.scaledToHeight(height)
            self.graphics_view.setPixmap(scaled_pic)
            self.prediction_label.setText(
                "Klasse: %s  mit  %.2f %% Wahrscheinlichkeit" % (response['classname'], response['prediction']))
            self.counter += 1

        except:
            print("No more Files")

    def __convert_img_to_qt(self, img):
        height, width, byte_value = img.shape
        byte_value = byte_value * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        mQImage = QImage(img, width, height, byte_value, QImage.Format_RGB888)
        return QPixmap.fromImage(mQImage)

    def multi_file(self):
        self.filePathArray.pop(0)
        for file in self.filePathArray:
            self.single_file()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    PicturesViewWindow = QtWidgets.QWidget()
    ui = UiPicturesViewWindow()
    ui.setupUi(PicturesViewWindow)
    PicturesViewWindow.show()
    sys.exit(app.exec_())
