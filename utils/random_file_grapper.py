import os
import math
import random
import shutil
from sklearn.utils import shuffle


class RandomFileGrapper:

    def __init__(self, PathFrom, PathTo, TrainPercent, ValPercent, TestPercent):
        self.pathfrom = PathFrom
        self.pathTo = PathTo
        self.trainPercent = TrainPercent / 100
        self.valPercent = ValPercent / 100
        self.testPerncent = TestPercent / 100

    def _get_file_array(self, directories):
        return os.listdir(directories)

    def count_filesI(self, directories):
        num_files = len([f for f in os.listdir(directories)
                         if os.path.isfile(os.path.join(directories, f))])
        return num_files

    def __create_random_array(self, input_array):
        input_array = shuffle(input_array)

        train_length = int(len(input_array) * self.trainPercent)
        val_length = int(len(input_array) * self.valPercent)
        test_length = int(len(input_array) * self.testPerncent)

        train_array = input_array[:train_length]
        val_array = input_array[train_length:train_length + val_length]
        test_array = input_array[train_length + val_length:train_length + val_length + test_length]

        return train_array, val_array, test_array

    def _create_random_number_array(self, percent, max_num):
        random_number_array = []
        for x in range(0, math.floor(max_num * percent)):
            random_number_array.append(random.randint(1, max_num))

        return random_number_array

    def copy_file(self, file_name, new_file_name, path_from, sub_directory):
        source_path = os.path.join(path_from, file_name)
        path_to = os.path.join(path_from, sub_directory)
        dst_path = os.path.join(path_to, new_file_name)

        try:
            shutil.copy(source_path, dst_path)
        except Exception:
            print(Exception)
            with open('logfile.txt', 'a') as logf:
                logf.write(source_path + '\n')

    def create_folder(self, directory, sub_directory):
        try:
            os.stat(os.path.join(directory, sub_directory))
        except:
            os.mkdir(os.path.join(directory, sub_directory))

    def _copy_array(self, array_to_copy, sub_directory):
        for index, file in enumerate(array_to_copy, start=0):
            print("%s Copy File: %s ", index, file)
            new_file_name = str(index) + '_' + file
            self.copy_file(file, new_file_name, self.pathfrom, sub_directory)

    def make_aata(self):
        self.create_folder(self.pathfrom, 'Train')
        self.create_folder(self.pathfrom, 'Val')
        self.create_folder(self.pathfrom, 'Test')

        files_in_folder = self._get_file_array(self.pathfrom)
        train, val, test = self.__create_random_array(files_in_folder)
        self._copy_array(train, 'Train')
        self._copy_array(val, 'Val')
        self._copy_array(test, 'test')


if __name__ == "__main__":
    pathFrom = 'D:\\tmp\\test'
    patTo = 'D:\\tmp\\test'

    directories = [d for d in os.listdir(pathFrom)
                   if os.path.isdir(os.path.join(pathFrom, d))]
    for dir in directories:
        path = os.path.join(pathFrom, dir)
        print(dir)
        rndGrapper = RandomFileGrapper(path, path, 80, 10, 10)
        rndGrapper.make_aata()
