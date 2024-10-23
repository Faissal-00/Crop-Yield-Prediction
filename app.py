from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocesser = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/test')
def test():
    return "Test page is working!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Year = request.form['Year']
    average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
    pesticides_tonnes = request.form['pesticides_tonnes']
    avg_temp = request.form['avg_temp']
    Area = request.form['Area']
    Item = request.form['Item']

    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

    # Convert the predicted yield to a standard float value
    prediction_value = float(predicted_yield[0])  # Get the first element and convert it to float

    return render_template('index.html', prediction=prediction_value)  # Pass the float value to the template

if __name__ == '__main__':
    app.run(debug=True)