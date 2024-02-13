from pyannote.audio import Pipeline
import streamlit as st
import joblib
import numpy as np
import tempfile
import os
from pydub import AudioSegment


""" 
A class that splits the uploaded audio file,
performs speaker diarization,
then emotion detection 
and outputs both results.
"""
class SpeakerEmotionDetector:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')
        self.emotion_model = joblib.load('emotion_model.pkl')

    def split_audio(self, audio_file, segment_length=10):
        audio = AudioSegment.from_file(audio_file)
        segment_ms = segment_length * 1000
        num_segments = np.ceil(len(audio) / segment_ms).astype(int)
        segments = []
        for i in range(num_segments):
            start_time = i * segment_ms
            end_time = min((i + 1) * segment_ms, len(audio))
            segments.append(audio[start_time:end_time])
        return segments

    def process_segment(self, segment, tmp_file):
        segment.export(tmp_file.name, format='wav')
        diarization = self.pipeline(tmp_file.name, min_speakers=1, max_speakers=4)
        print(type(diarization), diarization)
        for segment, speaker_label in diarization.itertracks(yield_label=True):
            emotion = self.emotion_model.predict(segment.crop(filename=tmp_file.name))
            st.write(f"Speaker {speaker_label}: {emotion}")

    def process_audio(self, audio_file):
        segments = self.split_audio(audio_file)
        for i, segment in enumerate(segments):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                self.process_segment(segment, tmp_file)
                os.unlink(tmp_file.name)

# Streamlit UI
st.title('Speaker Emotion Detection')
sed = SpeakerEmotionDetector()
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing...")
    sed.process_audio(uploaded_file)
    st.write("Processing complete.")
