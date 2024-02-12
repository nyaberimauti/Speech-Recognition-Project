from pyannote.audio import Pipeline
import streamlit as st
import pickle
import joblib

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')



# Load the model
emotion_model = joblib.load('emotion_model.pkl')



# Function to perform speaker diarization and emotion detection
def process_audio(audio_file):
    # Perform speaker diarization
    diarization = pipeline(audio_file, min_speakers=1, max_speakers=4)
    # Iterate over each speaker segment
    for segment, speaker_label in diarization.itertracks(yield_label=True):
        speaker_audio = segment.crop(filename=audio_file)
        # Perform emotion detection on speaker's segment
        emotion = emotion_model.predict(speaker_audio)
        st.write(f"Speaker {speaker_label}: {emotion}")

# Streamlit UI
def main():
    st.title('Speaker Emotion Detection')

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        st.write("Processing...")
        process_audio(uploaded_file)
        st.write("Processing complete.")

if __name__ == "__main__":
    main()
