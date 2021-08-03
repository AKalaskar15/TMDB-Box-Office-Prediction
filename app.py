from flask import Flask, request, render_template,jsonify
import json
import numpy as np
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd


app = Flask(__name__)

def load_models():
    model = pickle.load(open("model/revenue_rf2.pkl", "rb"))
    return model


def predict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    # load model
    print(to_predict)
    model = load_models()
    prediction = model.predict(to_predict)
    response = json.dumps({'response': str(prediction)})
    print(prediction)
    return prediction

@app.route('/')
@cross_origin()
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        print(to_predict_list)
        result = predict(to_predict_list)
        result = round(result[0],2)
        return render_template("result.html", prediction_text="Revenue prediction for these attributes is: Rs. {}".format(result))
    else:
        return render_template("index.html")
    return render_template("result.html")



