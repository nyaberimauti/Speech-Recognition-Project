import streamlit as st
from pyannote.audio import Pipeline
import assemblyai as aai
import tempfile
from textblob import TextBlob
import librosa
import numpy as np
import joblib


emotion_model = joblib.load('emotion_model.pkl')

def preprocess_audio(audio_file):
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=None)

        # Trim leading and trailing silence
        audio = librosa.effects.trim(audio)[0]

         # Normalize audio
        audio = librosa.util.normalize(audio)

        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Calculate other features (pitch, energy, spectral centroid, spectral bandwidth, spectral contrast)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches)
        energy = np.mean(librosa.feature.rms(y=audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=100.0, n_bands=6))

        # Flatten MFCCs to ensure consistent shape
        mfccs_flat = mfccs.flatten()

        # Combine all features into a single 1D array
        features_combined = np.hstack((mfccs_flat, pitch, energy, spectral_centroid, spectral_bandwidth, spectral_contrast))

        # Pad features_combined to ensure consistent shape
        max_length = len(features_combined)
        target_length = 2800  # Adjust as needed
        if max_length < target_length:
            padding_width = target_length - max_length
            features_combined = np.pad(features_combined, (0, padding_width), mode='constant')

        # Apply TruncatedSVD for dimensionality reduction
        features_reduced = np.array(list(features_combined))

        return features_reduced

def emotion_detection():
        # Check if the uploaded file is not None
        st.title("Emotion Detection")
        st.write("Emotion detection is about trying to detect the most prevalent emotion in a speech.")

        # File uploader
        st.sidebar.title("Upload Audio File")
        audio_file = st.sidebar.file_uploader("Upload WAV file", type=["wav"])
        
        if audio_file is None:
            st.write("No audio file uploaded.")
            return
        
        st.audio(audio_file, format='audio/wav')
        # Preprocess the audio file
        processed_features = preprocess_audio(audio_file)

        # Truncate the processed features if its size exceeds 100 dimensions
        if processed_features.size > 100:
            processed_features = processed_features[:100]

        # Reshape the processed features to have at most 100 dimensions
        num_columns = min(100, processed_features.size)
        num_rows = -1 if processed_features.size % num_columns == 0 else -(processed_features.size // num_columns + 1)
        processed_features = processed_features.reshape(num_rows, num_columns)

        # Perform emotion detection on the processed features
        emotion = emotion_model.predict(processed_features)

        my_string = ' '.join(emotion)
        
        # Display the detected emotion
        st.write(f"Emotion detected is: {my_string}")

# Function for diarization
def diarize(token):

    pipeline_diarize = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token)
    st.title("Speaker Diarization")
    st.write("Speaker diarization is the process of segmenting and labeling an audio recording into distinct segments, each corresponding to a different speaker.")

    # File uploader
    st.sidebar.title("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Upload WAV file", type=["wav"])
    

    if uploaded_file is not None:
        # Perform diarization on the uploaded audio file
        diarization_result = pipeline_diarize(uploaded_file)
        st.audio(uploaded_file, format='audio/wav')

        # Display the diarization results
        st.header("Diarization Results")
        st.write("Below are the speaker segments identified in the audio:")
        
        speakers = set()  # To store unique speakers
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            st.write(f"{speaker}: starts at {turn.start:.1f}s and ends at {turn.end:.1f}s")
            speakers.add(speaker)  # Add speaker to the set of unique speakers
        
        # Summary of unique speakers
        num_speakers = len(speakers)
        st.write(f"\nSummary: There were {num_speakers} unique speakers identified in the entire document.")

def get_file_path(uploaded_file):
    if uploaded_file is not None:
        # Use tempfile to create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Write the contents of the uploaded file to the temporary file
        temp_file.write(uploaded_file.read())

        # Close the temporary file to release the resources
        temp_file.close()

        # Return the file path of the temporary file
        return temp_file.name
    else:
        return None

# Function for audio trancription
def transcribe(token):
    aai.settings.api_key = token
    transcriber = aai.Transcriber()
    # Transcription code here
    st.title("Audio Transcription")
    st.write("Here we will transcript audio for text and do sentiment analysis based on the text.")
    # File uploader
    st.sidebar.title("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Upload WAV file", type=["wav"])
    st.audio(uploaded_file, format='audio/wav')

    if uploaded_file is not None:
        # Perform diarization on the uploaded audio file
        file_path = get_file_path(uploaded_file)
        transcript = transcriber.transcribe(file_path)

        # Display the diarization results
        st.header("Transcription Results")
        st.write("Below is the text for the transcribed audio:")
        st.write(transcript.text)

        blob = TextBlob(transcript.text)
    
        # Perform sentiment analysis
        sentiment_score = blob.sentiment.polarity
    
        # Determine sentiment label based on the polarity score
        if sentiment_score > 0:
            sentiment_label = "Positive"
        elif sentiment_score < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

    # Display sentiment score and label
        st.header("Sentiment Analysis")
        st.write("Sentiment Label:", sentiment_label)
    

# Define the Streamlit app
def main():

    st.title("Vibe Scribe's Speech Analysis App")
    st.write("Welcome to Vibe Scribe's Speech Analysis App. This app provides functionalities for Emotion Detection, Speaker Diarization, Audio Transcription and Sentiment Analysis.")
    st.sidebar.title("API Key Input")

    # Pyannote API Key Input
    st.sidebar.subheader("Pyannote API Key Input")
    pyannote_api_key = st.sidebar.text_input("Enter your Pyannote API key", type="password")
    if st.sidebar.button("Submit Pyannote API Key"):
        if pyannote_api_key:
            st.sidebar.success("Pyannote API key submitted successfully")
            # Call functions that use the Pyannote API key here
        else:
            st.sidebar.error("Please enter your Pyannote API key")

    # AssemblyAI API Key Input
    st.sidebar.subheader("AssemblyAI API Key Input")
    assembly_api_key = st.sidebar.text_input("Enter your AssemblyAI key", type="password")
    if st.sidebar.button("Submit AssemblyAI API Key"):
        if assembly_api_key:
            st.sidebar.success("AssemblyAI API key submitted successfully")
            # Call functions that use the AssemblyAI API key here
        else:
            st.sidebar.error("Please enter your AssemblyAI API key")

    st.sidebar.title("Select Functionality")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Emotion Detection", "Speaker Diarization", "Audio Transcription"])

    if app_mode == "Emotion Detection":
        emotion_detection()
    elif app_mode == "Speaker Diarization":
        diarize(pyannote_api_key)
    elif app_mode == "Audio Transcription":
        transcribe(assembly_api_key)

# Run the app
if __name__ == "__main__":
    main()
