import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

#Load the trained model
regmodel=pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    #the data will be in json format ie key value pairs, so have to 
    #extract the vauues and reshape it after converting into np array
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    #output is a two dimensional array, to get the prediction, take the first value
    print(output[0])
    #return the output in json format
    return jsonify(output[0])

if __name__ == "__main__": 
    app.run(debug=True)