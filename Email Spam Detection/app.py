import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

# loading the model
logmodel = pickle.load(open("logmodel.pkl", "rb"))

# loading the vectorizer
pickled_cv = pickle.load(open("cv.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    data = data["data"]
    print(data)
    vect = pickled_cv.transform(data).toarray()
    my_prediction = logmodel.predict(vect)
    print(my_prediction)
    return jsonify(my_prediction.tolist())


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("email")  # Get the specific form field 'email'
    print(message)
    vect = pickled_cv.transform([message]).toarray()  # Ensure message is in a list
    my_prediction = logmodel.predict(vect)
    print(my_prediction)
    return render_template(
        "home.html", prediction_text="The message is {}".format(my_prediction[0])
    )


if __name__ == "__main__":
    app.run(debug=True)
