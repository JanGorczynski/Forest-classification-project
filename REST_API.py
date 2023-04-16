from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import dill

#This code makes tf-gpu work
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Function that takes data as json and returns numpy array
def JSON_to_np(json):
    keys = list(json.keys())
    instance = [json[i] for i in keys] 
    return np.array(instance)

#List of all the forest types in our taks
forest_types = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']

#Loading all the models
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/h_class.pkl', 'rb') as f:
    HeuristicModel = dill.load(f)
with open('saved_models/heuristic.pkl', 'rb') as f:
    h_model = pickle.load(f)
with open('saved_models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('saved_models/dtree_model.pkl', 'rb') as f:
    dtree_model = pickle.load(f)
ann_model = load_model('saved_models/neural_network.h5')

#Initializing api
app = Flask(__name__)
api = Api(app)

#Class Heuristic that for get request with json that has all parameters returns predictions using heuristic model
class Heuristic(Resource):
    def get(self):
        print("1")
        request_data = request.json
        data = JSON_to_np(request_data)
        print("2")
        result = h_model.predict_one(data)
        f_type = forest_types[result-1]

        return {'model': 'Heuristic model','result': result,'forest type': f_type}
        

api.add_resource(Heuristic,'/heuristic')

#This class returns prediction using KNN model
class Knn(Resource):
    def get(self):
        request_data = request.json
        data = JSON_to_np(request_data)
        data = scaler.transform([data])
        result = knn_model.predict(data)
        result = result[0]
        f_type = forest_types[result-1]
        return {'model': 'k nearest neighbor','result': int(result),'forest type': f_type}

api.add_resource(Knn,'/knn')

#This class returns prediction using decision tree model
class DTree(Resource):
    def get(self):
        request_data = request.json
        data = JSON_to_np(request_data)
        data = scaler.transform([data])
        result = dtree_model.predict(data)
        result = result[0]
        f_type = forest_types[result-1]
        return {'model': 'Decision tree','result': int(result),'forest type': f_type}

api.add_resource(DTree,'/dtree')

#This class returns prediction using neural network
class NeuralNetwork(Resource):
    def get(self):
        request_data = request.json
        data = JSON_to_np(request_data)
        data = scaler.transform([data])
        pred = ann_model.predict(data.reshape(1,54))
        result = result = np.argmax(pred[0]) + 1
        f_type = forest_types[result-1]
        return {'model': 'Artificial neural network','result': int(result),'forest type': f_type}

api.add_resource(NeuralNetwork,'/ann')

if __name__ == '__main__':
    app.run(debug=False)