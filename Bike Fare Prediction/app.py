import flask
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import render_template

sc = StandardScaler()

# Use pickle to load in the pre-trained model
with open('bike_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__)

# Set up the main route
@app.route("/", methods=["GET", "POST"])
def main():
    if flask.request.method == "GET":
        # Just render the initial form, to get input
        return render_template("index.html")
    
    if flask.request.method == "POST":
        # Extract the input
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return render_template('index.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)    