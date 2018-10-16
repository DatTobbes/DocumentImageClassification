from wand.image import Image, Color
import os
import cv2
from win32com import client

class ImageConverter:

    def convert_singel_page_pdf(self, filename, size_X=150, size_Y= 150, doResize=False):
        """
        Converts a pdf to single Page PNG
        :param filename: file to convert
        :param size_X: image width in Pixel
        :param size_Y: image height in Pixel
        :param doResize:
        :return:
        """
        with Image(Image(filename=filename+"[0]")) as img:
            img.format = 'png'
            img.background_color = Color('white')
            img.alpha_channel = 'remove'
            image_filename = os.path.splitext(os.path.basename(filename))[0]
            image_filename = image_filename+'.png'
            image_filename = os.path.join(os.path.split(filename)[0],image_filename)
            if doResize:
                img.resize(size_X,size_Y)
            img.save(filename=image_filename)
        return image_filename

    def convert_singel_page_tiff(self, filename, size_X=150, size_Y= 150, do_resize=False):
            with Image(Image(filename=filename+"[0]")) as img:
                img.format = 'png'
                image_filename = os.path.splitext(os.path.basename(filename))[0]
                image_filename = image_filename+'.png'
                image_filename = os.path.join(os.path.split(filename)[0], image_filename)
                if do_resize:
                    img.resize(size_X,size_Y)
                img.save(filename=image_filename)
            return image_filename

    def __convert_word_to_pdf(self, file):
        """
        Converts a Wordfile to PDF
        :param file: Input Filename
        :return: the filename of the converted PDF
        """
        try:
            word = client.DispatchEx("Word.Application")
            if file.endswith(".docx"):
                    new_name = file.replace(".docx", r".pdf")
            if file.endswith(".doc"):
                    new_name = file.replace(".doc", r".pdf")
            new_file = new_name
            doc = word.Documents.Open(file)
            doc.SaveAs(new_file, FileFormat=17)
            doc.Close()
        except Exception:
            print(Exception)
        finally:
            word.Quit()
            return new_file

    def convertAllFiles(self, ParentDictionary):
        """
        Converts all files in a given directory to
        PNG

        :param ParentDictionary:
        :return:
        """
        for root, dirs, files in os.walk(ParentDictionary, topdown=False):
            for name in files:
                print(os.path.join(root, name))
                if 'Thumbs' in name:
                    continue
                print(name)
                try:
                    self.convert_document_to_image(os.path.join(root, name), root)
                except Exception:
                    print(Exception)
                    with open('logfile.txt', 'a') as logf:
                        logf.write(os.path.join(root, name) + '\n')

    def __convert_document_to_image(self, file_name):
        """
        Checks the Fileending and call the converterfunction
        :param file_name:
        :return:
        """
        if ".doc" in file_name:
            file_name= self.__convert_word_to_pdf(file_name)
            img=self.convert_singel_page_pdf(file_name, 150, 150, False)
            os.remove(file_name)
        elif ".pdf" in file_name:
            img=self.convert_singel_page_pdf(file_name, 150, 150, False)
        elif ".tif" in file_name:
            img=self.convert_singel_page_tiff(file_name, 150, 150, False)
        return img

    def return_document_image_file(self, file):
        return self.__convert_document_to_image(file)

    def load_document_image(self, file):
        """
        Interfacefuntion to convert a given document to a PNG
        :param file:
        :return:
        """
        image_file= self.__convert_document_to_image(file)
        img= cv2.imread(image_file)
        os.remove(image_file)
        return img



if __name__ == "__main__":

    converter= ImageConverter()
    img= converter.load_document_image('C:\\tmp\\test\MusterRechnung.pdf')

    cv2.imshow("test", img)
    cv2.imshow('image', img)
    cv2.waitKey(0)



