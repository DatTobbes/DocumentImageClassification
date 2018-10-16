
from flask import request
from classifier.multilayer_classifier import MultilayerClassifier
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import flask
from marshmallow import pprint
from PredictionSchema import Prediction, PredictionSchema

def __prediction_to_class(arg):
    options = {0: "Eingangsrechnungen",
               1: "Leistungsnachweise_LN_sy",
               2: "Ausgangsrechnungen",
               3: "Leistungsnachweise_LN_md",
               4: "Leistungsbeschreibung_LB",
               5: "Angebote"
               }
    return options.get(arg, "nothing")

app = flask.Flask(__name__)
model = None

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

def load_model():
    global classifier
    classifier=MultilayerClassifier()



@app.route("/getClasses", methods=["GET"])
def get_classnames():
    """
    :return: returns all classnames to Client
    """
    classnames=[]
    if flask.request.method == "GET":
        for i in range(5):
            classnames.append(__prediction_to_class(i))

    return flask.jsonify(classnames)


@app.route("/classify", methods=["POST"])
def predict():
    """
    takes a Documentimage from a client and calls the
    classifier to classify the image
    :return: the classnames, classprobability and class as a number
    """
    if flask.request.method == "POST":

        image =  request.json.get('data')

        image= np.asarray(image, dtype=np.float32)
        imageArray = image / 255

        predictions = classifier.classify_single_document(imageArray)
        prediction= Prediction(prediction=predictions['prediction'] * 100,classname=__prediction_to_class(predictions['class']),predicted_class=int(predictions['class']) )
        schema=PredictionSchema()
        result = schema.dump(prediction)
        pprint(result)

    return flask.jsonify(result)

if __name__ == "__main__":
    print(("* Loading PretrainedClassifier and starting Server..."
        "please wait until Server has fully started"))
    load_model()
    app.run(use_reloader=False, debug=True)