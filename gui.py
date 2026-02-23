import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib

fs = 44100
seconds = 5

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return np.hstack([mfcc_mean, zcr, flatness, rolloff])

def start_test():
    status.configure(text="🎤 Recording...")
    app.update()

    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("test.wav", fs, audio)

    model = joblib.load("fraud_model.pkl")
    feat = extract_features("test.wav")
    pred = model.predict([feat])[0]

    if pred == 0:
        result.configure(text="✅ Genuine Human Voice", text_color="green")
    elif pred == 1:
        result.configure(text="🚨 Replay Attack Detected", text_color="orange")
    else:
        result.configure(text="🤖 AI-Generated Voice", text_color="red")

app = ctk.CTk()
app.title("Voice Fraud Detection System")
app.geometry("500x420")

ctk.CTkLabel(app, text="🔐 Voice Fraud Detection",
             font=("Segoe UI", 20, "bold")).pack(pady=20)

ctk.CTkButton(app, text="Start Voice Test",
              width=260, height=45,
              command=start_test).pack(pady=30)

status = ctk.CTkLabel(app, text="Status: Idle")
status.pack(pady=10)

result = ctk.CTkLabel(app, text="", font=("Segoe UI", 16, "bold"))
result.pack(pady=40)

ctk.CTkLabel(app, text="Final Year Project | Cyber Security + AI",
             font=("Segoe UI", 10)).pack(side="bottom", pady=15)

app.mainloop()
