import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetes_prediction_2aug.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexf.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    if output ==0:
        
        return render_template('indexf.html', prediction_text='Patient is $ {}'.format('Non Diabetic'))
    else:
        return render_template('indexf.html', prediction_text='Patient is $ {}'.format('Diabetic'))


if __name__ == "__main__":
    app.run(debug=True)