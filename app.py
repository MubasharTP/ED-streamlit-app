import os
import soundfile as sf
from base64 import b64decode
import whisper
import openai
from transformers import pipeline
import streamlit as st

# Set OpenAI API key (replace with your actual API key)
openai.api_key = 'YOUR_API_KEY'

# Initialize Whisper model
model = whisper.load_model("large")

# Sentiment analysis function using GPT-4
def sentiment_analysis(text):
    """Analyze sentiment from the transcribed text using GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[{"role": "system", "content": "You are an assistant that analyzes the sentiment of texts."},
                      {"role": "user", "content": f"Analyze the sentiment of this text: {text}"}],
            max_tokens=60,
            temperature=0.7
        )
        sentiment = response['choices'][0]['message']['content'].strip()
        return sentiment
    except Exception as e:
        return f"Error during sentiment analysis: {str(e)}"

# Emotion detection pipeline using Hugging Face
emotion_pipeline = pipeline(
    "audio-classification",
    model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
)

def convert_to_wav(input_file, output_file):
    """Convert audio from webm to wav format."""
    os.system(f"ffmpeg -i {input_file} -ar 16000 -ac 1 {output_file}")

def transcribe_audio(file_path):
    """Transcribes audio using OpenAI's Whisper model."""
    try:
        result = model.transcribe(file_path)
        return result['text']
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def emotion_analysis(audio_file_path):
    """Analyze emotion from the audio using a pre-trained model."""
    try:
        audio_data, sample_rate = sf.read(audio_file_path)
        emotion_prediction = emotion_pipeline(audio_data)
        return emotion_prediction
    except Exception as e:
        return f"Error in emotion analysis: {str(e)}"

# Streamlit UI
st.title("Audio Sentiment & Emotion Analysis")

st.subheader("Upload or Record Audio")
audio_file = st.file_uploader("Choose an audio file (mp3, wav, webm)", type=["mp3", "wav", "webm"])

if audio_file:
    st.audio(audio_file, format="audio/webm")
    
    # Save the uploaded audio file to a local directory
    with open("uploaded_audio.webm", "wb") as f:
        f.write(audio_file.getbuffer())

    # Convert to WAV
    convert_to_wav("uploaded_audio.webm", "audio.wav")

    # Transcribe the audio file
    transcription = transcribe_audio("audio.wav")
    st.write("Transcription:", transcription)

    # Perform Sentiment Analysis
    sentiment = sentiment_analysis(transcription)
    st.write("Sentiment Analysis:", sentiment)

    # Perform Emotion Analysis
    emotion = emotion_analysis("audio.wav")
    st.write("Emotion Analysis:", emotion)

else:
    st.write("Please upload an audio file to analyze.")
