from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
import yaml

class Model:
    ''' This class contains the scaler for the data
    and the model to be used to evaluate the data'''

    def __init__(self, model_conf):
        self.conf = yaml.load(open(model_conf))
        self.variables = self.conf["variables"]
        self.model = load_model(self.conf["model_path"])
        self.scaler = joblib.load(self.conf["scaler"])

    def evaluate(self, data, batchsize):
        # transform the data 
        data_res = self.scaler.transform(data)
        return self.model.predict(data_res, batch_size=batchsize)
    