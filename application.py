import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
first_predict = pickle.load(open("first_predict.pkl", "rb"))
second_predict = pickle.load(open("second_predict.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("website.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    labels = ['Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
              'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'PortScan']
    float_features = [float(x) for x in request.form.values()]
    print(float_features)
    scaled_data = scaler.fit_transform(float_features)
    features = [np.array(float_features)]
    encoded = encoder.predict(features)
    prediction1 = first_predict.predict(encoded)
    if prediction1 == 0:
        final_prediction = "BENIGN"
    else:
        final_prediction = labels[second_predict.predict(encoded)]

    return render_template("website.html", prediction_text="Website activity is {}".format(final_prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)
