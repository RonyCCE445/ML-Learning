from flask import Flask, jsonify, request, render_template
import pickle
from flask_cors import CORS
import numpy as np

#loading the model
with open("iris_model.pkl", "rb") as f:
    model, class_names = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data=request.get_json()
    input_data= np.array([
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    predicted_class = class_names[prediction]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)