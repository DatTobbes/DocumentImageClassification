import sys
import os
import shutil
import json

PY3K = sys.version_info >= (3,)
if PY3K:
    from tkinter import filedialog as fd
else:
    from Tkinter import tkFileDialog as fd

categories = {0: "letter",
              1: "form",
              2: "email",
              3: "handwritten",
              4: "advertisement",
              5: "scientific report",
              6: "scientific puplication",
              7: "specification",
              8: "file folder",
              9: "news articel",
              10: "budget",
              11: "invoice",
              12: "presentation",
              13: "questionnaire",
              14: "resume",
              15: "memo"
              }


def number_to_class(arg, Options):
    options = Options
    return options.get(arg, "nothing")


filename = fd.askopenfilename()
baseFileName = os.path.basename(filename).split(".txt")[0]
labelFilePath = os.path.dirname(filename)
path = os.path.join(os.path.dirname(os.path.dirname(filename)), baseFileName)


def make_directories():
    if not os.path.exists(path):
        os.makedirs(path)

    for categorie in categories.values():
        directory = os.path.join(path, categorie)

        if not os.path.exists(directory):
            os.makedirs(directory)


make_directories()
error_dict = {}
error_array = []
line_dict = {}


def process(line):
    a = line.split(" ")
    line_dict["path"] = labelFilePath + "/" + a[0]
    line_dict["class"] = number_to_class(int(a[1]), categories)
    try:
        copy_img(line_dict["path"], path, line_dict["class"])
    except (FileNotFoundError):
        print("file not found")
    except:
        e = sys.exc_info()[0]
        print(e)
        error_dict["file"] = line_dict["path"]
        error_array.append(error_dict)


def copy_img(from_dir, to_dir, category):
    copy_from = from_dir
    copy_to = os.path.join(to_dir, category)
    shutil.move(copy_from, copy_to)


with open(filename) as f:
    counter = 0
    for line in f:
        line = line.replace('\n', '')
        line = line.replace('\t', '')
        process(line)
        counter += 1
        print(str(counter))

    with open('data.json', 'a') as outfile:
        outfile.writelines(json.dumps(item) + '\n' for item in error_array)
