from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import librosa
import pickle

app = FastAPI()

# Load model
model = tf.keras.models.load_model("lstm_audio_model.keras", compile=False)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# MFCC extraction
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T
    
    max_len = 216
    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:max_len]
    
    return np.expand_dims(mfcc, axis=0)

# API endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(contents)

    features = extract_mfcc("temp.wav")

    prediction = model.predict(features)
    index = np.argmax(prediction)

    return {"prediction": le.classes_[index]}