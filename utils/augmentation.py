import cv2
import numpy as np
import os
from sklearn.utils import shuffle


class ImageAugmentation:

    def rnd_flip(self, image):
        image = cv2.flip(image, 0)
        return image

    def rnd_rotation(self, image):
        img = image
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        return img_rotation

    def _get_file_array(self, directory):
        return os.listdir(directory)

    def create_random_array(self, directory, percent):
        input_array = self._get_file_array(directory)
        input_array = shuffle(input_array)
        length = int(len(input_array) * percent)
        return input_array[:length]

    def rnd_shear(self, shear_range, img):
        rows, cols, ch = img.shape
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, shear_M, (cols, rows))

        return img


if __name__ == "__main__":

    augmenter = ImageAugmentation()
    dir = 'C:\\tmp\Images'

    directories = [d for d in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, d))]

    for d in directories:
        new_dir = os.path.join(dir, d)
        files = augmenter.create_random_array(new_dir, 0.2)
        for file in files:
            new_file_name = 'A_' + file
            img = cv2.imread(os.path.join(new_dir, file))
            img = augmenter.rnd_rotation(img)
            cv2.imwrite(os.path.join(new_dir, new_file_name), img)
