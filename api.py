from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import librosa

app = Flask(__name__)

# Load trained model ONCE

model = joblib.load("fraud_model.pkl")


import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # 1–13 MFCCs (mean)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 14 Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 15 Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    # 16 Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    features = np.hstack([
        mfccs_mean,
        zcr,
        chroma,
        centroid
    ])

    return features




@app.route("/predict", methods=["POST"])
def predict():
    temp_path = "temp.wav"

    with open(temp_path, "wb") as f:
        f.write(request.data)

    features = extract_features(temp_path).reshape(1, -1)
    pred = model.predict(features)[0]

    os.remove(temp_path)

    if pred == 0:
        result = "Genuine Human Voice"
    elif pred == 1:
        result = "Replay Attack Detected"
    else:
        result = "AI Generated Voice"

    return jsonify({"result": result})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
