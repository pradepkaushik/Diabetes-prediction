import numpy as np
import pandas as pd
from flask_cors import  cross_origin
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@cross_origin()
@app.route('/')
def home():
    return render_template('indexf.html')
@cross_origin()
@app.route('/predict',methods=['POST'])
def predict():
    Age = int(request.form['Age'])
    Gender = int(request.form['Gender'])
    Polyuria = int(request.form['Polyuria'])
    Polydipsia = int(request.form['Polydipsia'])
    sudden weight loss = int(request.form['sudden weight loss'])
    weakness = int(request.form['weakness'])
    Polyphagia = int(request.form['Polyphagia'])
    visual blurring = int(request.form['visual blurring'])
    Irritability = int(request.form['Irritability'])
    partial paresis = int(request.form['partial paresis'])
    Alopecia = int(request.form['Alopecia'])

    filename = 'diabetes_prediction_2aug.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    #scalar = pickle.load(open("sandardScalar.sav", 'rb'))
    # predictions using the loaded model file
    prediction = loaded_model.predict(
        [[Age, Gender, Polyuria, Polydipsia,sudden weight loss,weakness,Polyphagia,
          visual blurring,Irritability,partial paresis,Alopecia]])
    if prediction ==[1]:
            prediction = "diabetes"

    else:
            prediction = "Non diabetic"

    # showing the prediction results in a UI
    if  prediction =="diabetes":

        return render_template('diabetes.html', prediction=prediction)
    else:
        return render_template('non diabetes.html',prediction=prediction)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
	#app.run(debug=True)