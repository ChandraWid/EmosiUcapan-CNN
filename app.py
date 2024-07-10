import numpy as np
import librosa
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model once when the app starts
@st.cache_resource
def load_trained_model():
    return load_model('ucapanmodel.h5')

# Define the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

# Function to preprocess audio
def preprocess_audio(path, sr=16000, augment=False):
    y, sr_original = librosa.load(path, sr=sr)
    y = librosa.effects.preemphasis(y)
    yt, _ = librosa.effects.trim(y)
    yt = librosa.util.normalize(yt)
    return yt, sr

# Function to extract MFCC features
def extract_features(path, sr=16000, n_mfcc=13, hop_length=512):
    data, sr = preprocess_audio(path, sr=sr)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Streamlit app
st.title("Pengenalan Emosi Ucapan")

# Load the model
model = load_trained_model()

# Multiple file input
uploaded_files = st.file_uploader("Masukkan File Audio", type=["wav"], accept_multiple_files=True)

# Add a button to trigger predictions
if st.button('Prediksi'):
    if uploaded_files:
        # Initialize lists to store results
        audio_names = []
        predictions = []
        original_audio_data = []

        for uploaded_file in uploaded_files:
            audio_name = uploaded_file.name
            audio_names.append(audio_name)

            # Save the uploaded file temporarily
            with open(audio_name, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Extract features and predict
            mfccs = extract_features(audio_name)
            mfccs = mfccs.reshape(1, mfccs.shape[0], 1)
            y_pred = model.predict(mfccs)
            y_pred_label = np.argmax(y_pred, axis=1)
            predicted_emotion = label_encoder.inverse_transform(y_pred_label)[0]

            predictions.append(predicted_emotion)

            # Preprocess audio for playback
            original_audio, sr = preprocess_audio(audio_name)
            original_audio_data.append((original_audio, sr))

        # Display audio players and results in a table
        st.write("Hasil Prediksi:")
        for i, (audio_name, pred) in enumerate(zip(audio_names, predictions)):
            st.write(f"Audio File: {audio_name}")
            st.audio(original_audio_data[i][0], format='audio/wav', sample_rate=original_audio_data[i][1], start_time=0)
            st.write(f"Prediksi: {pred}")
            st.write("")

        # Create DataFrame for results
        results_df = pd.DataFrame({
            'File Audio': audio_names,
            'Prediksi': predictions
        })

        st.write("Tabel Hasil Prediksi:")
        st.write(results_df)
    else:
        st.error("Harap unggah setidaknya satu file audio.")
