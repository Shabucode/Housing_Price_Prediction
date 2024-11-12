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

# Feature names used in the training model
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'MedInc': float(request.form['MedInc']),
        'HouseAge': float(request.form['HouseAge']),
        'AveRooms': float(request.form['AveRooms']),
        'AveBedrms': float(request.form['AveBedrms']),
        'Population': float(request.form['Population']),
        'AveOccup': float(request.form['AveOccup']),
        'Latitude': float(request.form['Latitude']),
        'Longitude': float(request.form['Longitude']),
    }
    # Convert the list into a DataFrame with correct feature names
    data_df = pd.DataFrame([data])
    new_data = scalar.transform(np.array(data_df).reshape(1,-1))
    output = regmodel.predict(new_data)[0]
    return render_template('home.html', predicted_text = "The House Price Predicted is {}".format(output))

if __name__ == "__main__": 
    app.run(debug=True)