import streamlit as st
import os
import time
import tempfile
import librosa
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.audio_processing import extract_features, load_audio
from utils.model_training import load_or_train_model
from utils.visualization import plot_waveform, plot_spectrogram, plot_confidence

# Set page configuration
st.set_page_config(
    page_title="TuneType: Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'genre_labels' not in st.session_state:
    st.session_state.genre_labels = None
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None

def main():
    # Display header
    st.title("🎵 TuneType: Genre Classifier")
    
    # Display welcome message
    st.markdown("""
    ### Welcome to TuneType!
    Upload your music and discover its genre!
    
    This app uses machine learning to classify music into genres based on audio features extracted from the sound.
    
    **Supported genres:**
    - Blues
    - Classical
    - Country
    - Disco
    - Hip-Hop
    - Jazz
    - Metal
    - Pop
    - Reggae
    - Rock
    
    For best results, upload a high-quality audio file with a clear, single genre.
    """)
    
    # Load or train the model
    with st.spinner("Loading model..."):
        model, genre_labels = load_or_train_model()
        st.session_state.model = model
        st.session_state.genre_labels = genre_labels
    
    # File uploader
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
    
    # Process uploaded file
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load audio file
        try:
            with st.spinner("Loading audio file..."):
                audio_data, sr = load_audio(tmp_path)
                st.session_state.audio_data = audio_data
                st.session_state.sr = sr
                st.session_state.audio_file = tmp_path
            
            # Display audio player and file info
            st.subheader("Audio Preview")
            st.audio(tmp_path, format=f'audio/{os.path.splitext(uploaded_file.name)[1][1:]}')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"File: {uploaded_file.name}")
            with col2:
                st.info(f"Duration: {librosa.get_duration(y=audio_data, sr=sr):.2f} seconds")
            
            # Display waveform
            st.subheader("Waveform Visualization")
            waveform_fig = plot_waveform(audio_data, sr)
            st.plotly_chart(waveform_fig, use_container_width=True)
            
            # Analyze button
            if st.button("Analyze and Predict Genre", type="primary"):
                try:
                    # Extract features
                    with st.spinner("Extracting audio features..."):
                        features = extract_features(audio_data, sr)
                        st.session_state.features = features
                    
                    # Make prediction
                    with st.spinner("Predicting genre..."):
                        # Add artificial delay for better UX
                        time.sleep(1)
                        
                        # Verify feature dimensions match model expectations
                        expected_features = 59
                        actual_features = features.shape[0]
                        
                        if actual_features != expected_features:
                            st.error(f"Feature mismatch: Expected {expected_features} features but got {actual_features}. Adjusting...")
                            # Adjust features to match expected dimensions
                            if actual_features < expected_features:
                                # Pad with zeros if we have too few
                                features = np.pad(features, (0, expected_features - actual_features), 'constant')
                            else:
                                # Truncate if we have too many
                                features = features[:expected_features]
                        
                        # Make prediction with adjusted features
                        prediction_idx, probabilities = model.predict([features]), model.predict_proba([features])[0]
                        
                        # Convert prediction index to genre name
                        genre_name = genre_labels[prediction_idx[0]]
                        st.session_state.prediction = genre_name
                        st.session_state.probabilities = probabilities
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.success(f"Predicted Genre: **{st.session_state.prediction.upper()}**")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please try a different audio file or retry with the same file.")
                    return
                
                # Display confidence levels if prediction was successful
                if 'probabilities' in st.session_state and st.session_state.probabilities is not None:
                    confidence_fig = plot_confidence(st.session_state.probabilities, genre_labels)
                    st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Display spectrogram
                st.subheader("Spectrogram")
                spectrogram_fig = plot_spectrogram(audio_data, sr)
                st.plotly_chart(spectrogram_fig, use_container_width=True)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Background image with attribution - using a banner at the bottom
    st.markdown("""
    ---
    ### About
    
    TuneType uses librosa for audio feature extraction and scikit-learn for genre classification. The model is trained on the GTZAN dataset.
    
    *Banner images attribution: Photos from Unsplash by Marcela Laskoski, Clark Young, Gabriel Gurrola and Israel Palacio*
    """)

if __name__ == "__main__":
    main()
