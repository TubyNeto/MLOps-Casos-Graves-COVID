import os
import pickle
import numpy as np
import pandas as pd

import mlflow
from flask import Flask, request, jsonify


logged_model = f'C:/Users/Tuby Neto/Desktop/projeto-mlops/mlruns/1/08b0d8a965fd4419a79a8db773a31237/artifacts/model'

model = mlflow.pyfunc.load_model(logged_model)


# Preparação das Features

def prepare_features(ride):
    features = pd.json_normalize(ride)
    
    print(features)

    return features


# Predição 
def predict(features):

    preds = model.predict(features)

    return jsonify({'prediction': int(preds[0])})



app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    print(ride)
    features = prepare_features(ride)
  
    pred = predict(features)
    return pred 

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)