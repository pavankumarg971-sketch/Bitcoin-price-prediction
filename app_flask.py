from flask import Flask, jsonify
from predict import predict_rf, predict_lstm

app = Flask(__name__)

@app.route("/predict/rf")
def rf():
    return jsonify({"prediction": float(predict_rf())})

@app.route("/predict/lstm")
def lstm():
    return jsonify({"prediction": float(predict_lstm())})

if __name__ == "__main__":
    app.run(debug=True)
