# 🎧 LSTM-Based Audio Sequence Prediction System

## 📌 Project Overview

This project implements an **end-to-end AI system** using a **Long Short-Term Memory (LSTM)** model for audio-based sequence prediction.

The system:

* Takes audio input (.wav)
* Extracts features using MFCC
* Uses an LSTM model to learn temporal patterns
* Predicts the speaker/class
* Deploys the model using **FastAPI** for real-time inference

---

## 🎯 Objectives

* Develop an LSTM-based sequence prediction model
* Extract meaningful features from audio signals
* Perform multi-class classification of speakers
* Deploy the model using FastAPI
* Build a complete AI pipeline from data → deployment

---

## 📊 Dataset Information

### Dataset Name

Audio Classifier Dataset

### Source

Kaggle

### Dataset Link

https://www.kaggle.com/datasets/aklimarimi/audio-classifier-dataset

---

### 📖 Description

The dataset contains audio recordings of multiple motivational speakers such as:

* Oprah Winfrey
* Brené Brown
* Gary Vee
* Eric Thomas
* Simon Sinek
* Jay Shetty
* and others

Each class contains multiple `.wav` audio samples.

---

### ⚙️ Dataset Characteristics

* Format: `.wav` files
* Type: Audio classification (multi-class)
* Variable-length sequences
* Real-world speech data

---

## 🔧 Data Preprocessing

The following steps were performed:

1. Audio loading using `librosa`
2. Feature extraction using **MFCC (Mel-Frequency Cepstral Coefficients)**
3. Padding/truncation to fixed sequence length
4. Label encoding using `LabelEncoder`

---

## 🎼 MFCC Feature Explanation

MFCC converts audio signals into a numerical representation based on human hearing.

* Captures frequency characteristics
* Reduces noise and redundancy
* Converts audio → time-series features

Example shape:

```
(time_steps, features) = (216, 13)
```

---

## 🤖 Model Architecture

* Model: LSTM (Long Short-Term Memory)
* Input Shape: (216, 13)
* Layers:

  * LSTM (128 units)
  * Dense (64 units, ReLU)
  * Output Dense (Softmax)

---

## 🧠 LSTM Mathematical Explanation (Important for Viva)

LSTM consists of three gates:

### 1. Forget Gate

Decides what information to remove:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

### 2. Input Gate

Decides what new information to store:

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
```

### 3. Output Gate

Determines output:

```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```

### Cell State Update:

```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

👉 This allows LSTM to **learn long-term dependencies in sequences**

---

## 📈 Model Training

* Epochs: 20
* Batch Size: 32
* Loss Function: Categorical Crossentropy
* Optimizer: Adam

---

## 📊 Model Evaluation

* Train-Test Split: 80% / 20%
* Metrics Used:

  * Accuracy
  * Confusion Matrix
  * Classification Report

👉 Model shows good performance but some misclassification due to similar voice patterns.

---

## 💾 Model Saving

Model saved in modern Keras format:

```
lstm_audio_model.keras
```

Label encoder saved as:

```
label_encoder.pkl
```

---

## 🚀 Deployment (FastAPI)

The model is deployed using FastAPI.

### Endpoint:

```
POST /predict
```

### Functionality:

* Accepts `.wav` file
* Extracts MFCC features
* Returns predicted speaker

---

## ▶️ How to Run the Project

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd LSTM_project
```

### 2. Create Virtual Environment

```bash
python -m venv .env
.env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Server

```bash
uvicorn main:app --reload
```

### 5. Open Browser

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Testing

* Upload `.wav` file using Swagger UI
* Click **Execute**
* Get JSON response

Example:

```json
{
  "prediction": "Eric Thomas"
}
```

---

## 📸 Deployment Proof

Include screenshot of:

* Swagger UI
* File upload
* Output JSON

---

## 📁 Project Structure

```
LSTM_project/
│
├── main.py
├── lstm_audio_model.keras
├── label_encoder.pkl
├── requirements.txt
├── notebook.ipynb
└── README.md
```

---

## ⚠️ Limitations

* Similar voice tones may cause misclassification
* Model accuracy depends on dataset quality
* Real-world noise can affect predictions

---

## ✅ Conclusion

This project successfully demonstrates:

* Sequence learning using LSTM
* Audio feature extraction using MFCC
* Real-time prediction via FastAPI

It provides a complete pipeline from **data preprocessing → model training → deployment**

---

## 🚀 Expected Outcome

A fully functional AI system capable of:

* Processing audio input
* Predicting speaker class
* Providing real-time results via API

---
