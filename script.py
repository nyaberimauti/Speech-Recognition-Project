# from pyannote.audio import Pipeline
# import streamlit as st
# import joblib
# import numpy as np
# import tempfile
# import os
# from pydub import AudioSegment


# """ 
# A class that splits the uploaded audio file,
# performs speaker diarization,
# then emotion detection 
# and outputs both results.
# """
# class SpeakerEmotionDetector:
#     def __init__(self):
#         self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')
#         self.emotion_model = joblib.load('emotion_model.pkl')

#     def split_audio(self, audio_file, segment_length=10):
#         audio = AudioSegment.from_file(audio_file)
#         segment_ms = segment_length * 1000
#         num_segments = np.ceil(len(audio) / segment_ms).astype(int)
#         segments = []
#         for i in range(num_segments):
#             start_time = i * segment_ms
#             end_time = min((i + 1) * segment_ms, len(audio))
#             segments.append(audio[start_time:end_time])
#         return segments

#     def process_segment(self, segment, tmp_file):
#         segment.export(tmp_file.name, format='wav')
#         diarization = self.pipeline(tmp_file.name, min_speakers=1, max_speakers=4)
#         for segment, speaker_label in diarization.itertracks(yield_label=True):
#             emotion = self.emotion_model.predict(segment.crop(filename=tmp_file.name))
#             st.write(f"Speaker {speaker_label}: {emotion}")

#     def process_audio(self, audio_file):
#         segments = self.split_audio(audio_file)
#         for i, segment in enumerate(segments):
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
#                 self.process_segment(segment, tmp_file)
#                 os.unlink(tmp_file.name)

# # Streamlit UI
# st.title('Speaker Emotion Detection')
# sed = SpeakerEmotionDetector()
# uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

# if uploaded_file is not None:
#     st.audio(uploaded_file, format='audio/wav')
#     st.write("Processing...")
#     sed.process_audio(uploaded_file)
#     st.write("Processing complete.")


from pyannote.audio import Pipeline
import streamlit as st
import joblib
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import librosa
from sklearn.decomposition import TruncatedSVD


class SpeakerEmotionDetector:
    def __init__(self, sampling_rate = 16000, n_components = 100):
        self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')
        self.emotion_model = joblib.load('best_svm_model.pkl')
        self.sampling_rate = sampling_rate
        self.svd = TruncatedSVD(n_components=n_components)


    def split_audio(self, audio_file, segment_length=10):
        audio = AudioSegment.from_file(audio_file)
        audio_duration_secs = len(audio) / 1000  # Convert milliseconds to seconds

        # Check if audio duration is less than 25 seconds
        if audio_duration_secs < 25:
            return [(0, len(audio))]  # Return a single segment covering the entire audio

        segment_ms = segment_length * 1000
        num_segments = np.ceil(audio_duration_secs / segment_length).astype(int)
        segments = []
        for i in range(num_segments):
            start_time = i * segment_ms
            end_time = min((i + 1) * segment_ms, len(audio))
            segments.append((start_time, end_time))
        return segments

    def preprocess_audio(self, audio_file):
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
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))

        # Flatten MFCCs to ensure consistent shape
        mfccs_flat = mfccs.flatten()

        # Combine all other features into a single 1D array
        features_combined = np.hstack((mfccs_flat, pitch, energy, spectral_centroid, spectral_bandwidth, spectral_contrast))

        # Update max_length if necessary
        max_length = 2000  # Specify the maximum length
        
        # Pad the features if they are shorter than the maximum length
        if len(features_combined) < max_length:
            padding_width = max_length - len(features_combined)
            padded_features = np.pad(features_combined, (0, padding_width), mode='constant')

        features_reshaped = padded_features.reshape(1, -1)

        st.write("Shape of features_reshaped:", features_reshaped.shape)

        features_reduced = features_reshaped

        return features_reduced

    # def process_audio(self, audio_file):
    #     # Check if the uploaded file is not None
    #     if audio_file is None:
    #         st.write("No audio file uploaded.")
    #         return

    #     # Preprocess the audio file
    #     processed_features = self.preprocess_audio(audio_file)

    #     # Reshape the processed features to have a single sample
    #     i = int(len(processed_features) / 28)
    #     processed_features = processed_features.reshape(-1, i)

    #     # Perform emotion detection on the processed features
    #     emotion = self.emotion_model.predict(processed_features)

    #     # Display the detected emotion
    #     st.write(f"Emotion detected is: {emotion}")
    def process_audio(self, audio_file):
        # Check if the uploaded file is not None
        if audio_file is None:
            st.write("No audio file uploaded.")
            return

        # Preprocess the audio file
        processed_features = self.preprocess_audio(audio_file)

        st.write("Shape of processed_features:", processed_features.shape)

        # Perform emotion detection on the processed features
        emotion = self.emotion_model.predict(processed_features)


        # Display the detected emotion
        st.write(f"Emotion detected is: {emotion}")
    def speaker_diarization(self, audio_file):
        # Check if the uploaded file is not None
        if audio_file is None:
            st.write("No audio file uploaded.")
            return

        # Preprocess the audio file
        processed_features = self.preprocess_audio(audio_file)

        st.write("Shape of processed_features:", processed_features.shape)

        # Perform emotion detection on the processed features
        emotion = self.emotion_model.predict(processed_features)


        # Display the detected emotion
        st.write(f"Emotion detected is: {emotion}")
# Streamlit UI
st.title('Speaker Emotion Detection')
sed = SpeakerEmotionDetector()
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing...")
    sed.process_audio(uploaded_file)
    st.write("Done.")
