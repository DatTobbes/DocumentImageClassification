import marshmallow
import datetime as dt

from marshmallow import Schema, fields

class Prediction(object):
    def __init__(self, prediction, classname, predicted_class):
        self.prediction = prediction
        self.classname = classname
        self.predicted_class= predicted_class

class PredictionSchema(Schema):
    prediction = fields.Float()
    classname= fields.Str()
    predicted_class=fields.Str()
