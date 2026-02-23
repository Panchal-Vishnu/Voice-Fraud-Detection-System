import librosa
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def extract_features(file):
    y, sr = librosa.load(file, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    return np.hstack([mfcc_mean, zcr, flatness, rolloff])

X, y = [], []

labels = {
    "human": 0,
    "replay": 1,
    "ai": 2
}

for folder, label in labels.items():
    path = f"dataset/{folder}"
    for file in os.listdir(path):
        if file.endswith(".wav"):
            feat = extract_features(os.path.join(path, file))
            X.append(feat)
            y.append(label)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained & saved as fraud_model.pkl")
