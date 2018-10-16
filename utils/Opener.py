# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import os

"""
This Class is used to select Files and Directories to classify

"""


class Opener(QWidget):
    def __init__(self):
        super().__init__()
        self.filePathArray = []

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Images (*.png *.xpm *.jpg);; Text Files(*.doc *.docx *.pdf)",
                                                  options=options)
        if file_name:
            return file_name

    def get_root_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to save Documents")
        return directory

    def open_file_names_dialog(self):
        self.filePathArray = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Documents")
        if directory:
            for file in os.listdir(directory):
                self.filePathArray.append(directory + "/" + file)
        return self.filePathArray

    def get_documents_from_directory(self):
        self.filePathArray = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            for file in os.listdir(directory):
                if file.endswith(('.docx', '.pdf', '.tiff', '.doc')):
                    self.filePathArray.append(self.rename(os.path.join(directory, file)))

        return self.filePathArray

    def get_file_path_array(self):
        return self.filePathArray

    def rename(self, file_name):
        """
        removes German Umlaute in filenames
        """
        umlaut_dictionary = {u'Ä': 'Ae',
                             u'Ö': 'Oe',
                             u'Ü': 'Ue',
                             u'ä': 'ae',
                             u'ö': 'oe',
                             u'ü': 'ue'
                             }
        umap = {ord(key): (val) for key, val in umlaut_dictionary.items()}
        new_file_name = file_name.translate(umap)
        print(file_name)
        os.rename(file_name, new_file_name)
        return new_file_name


if __name__ == "__main__":
    op = Opener()
    op.get_documents_from_directory()
