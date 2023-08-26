import streamlit as st
import requests
import librosa
import moviepy.editor as mp
from moviepy.audio.AudioClip import AudioArrayClip
# import audioread
import numpy as np
from io import BytesIO
import tempfile
import pathlib
import json
import speech_recognition as sr
from scipy.io import wavfile
# import noisereduce as nr
# import os
from PIL import Image
import base64
# import pydub
import soundfile as sf
import pandas as pd

temp_dir = tempfile.TemporaryDirectory()




def extract_audio(video_file):
    # stringio = StringIO(video_file.getvalue().decode("unicode_escape"))
    # print('Started Extracting : ', video_file)
    video_clip = mp.VideoFileClip(str(video_file))



    # fps = video_clip.fps
    fps = 44100
    # print("FPS", fps)
    audio_array = video_clip.audio.to_soundarray(fps=fps)
    # a = clip.to_soundarray(nbytes=4, buffersize=1000, fps=fps)
    n = AudioArrayClip(audio_array, fps=fps)
    writing_file_path = pathlib.Path(temp_dir.name) / "audio.wav"
    # writing_file_path = pathlib.Path("/Users/yamuna/MSc/App") / "audio.wav"
    # n.write_audiofile(writing_file_path, codec = 'pcm_s16le')
    sf.write(str(writing_file_path), audio_array, 44100)
    speech_writing_file_path = pathlib.Path(temp_dir.name) / "audio_transcribe.wav"
    y = (np.iinfo(np.int32).max * (audio_array/np.abs(audio_array).max())).astype(np.int32)
    wavfile.write(speech_writing_file_path, fps, y)
    return writing_file_path

# As a first step, we extract only mfcc features
def extract_audio_features(uploaded_file):
    audio_loc = extract_audio(uploaded_file)

    audio_file = open(audio_loc,'rb') #enter the filename with filepath
    audio_bytes = audio_file.read() #reading the file
    st.audio(audio_bytes, format='audio/ogg') #displaying the audio

    y, sr = librosa.load(audio_loc)
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    # print('mfcc : ', mfcc)
    mfcc_mean = mfcc.mean(axis=1).T
    mfcc_std = mfcc.std(axis=1).T
    mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
    return list(mfcc_feature)

def extract_images(uploaded_file_path):
    output_folder = "img"
    # Open the video file
    video_clip = mp.VideoFileClip(str(uploaded_file_path))
    # Get the duration of the video in seconds
    video_duration = video_clip.duration

    # Get frames at regular intervals (e.g., 1 frame per second)
    frame_interval = 1  # Adjust this as needed
    frames = []
    base64_frames = []

    for t in range(0, int(video_duration), frame_interval):
        frame = video_clip.get_frame(t)
        base64_str = image_to_base64(frame)
        base64_frames.append(base64_str)
        frames.append(frame)

    st.write(f"Extracted {len(frames)} frames from the video:")

    # Display extracted frames
    cols = st.columns(len(frames))
    for idx, frame in enumerate(frames):
        cols[idx].image(Image.fromarray(frame), caption=f"Frame {idx + 1}")

    return base64_frames[0] if len(base64_frames) > 0 else base64_frames


def speech_to_text():
    print('start of speech')
    writing_file_path = pathlib.Path(temp_dir.name) / "audio_transcribe.wav"
    # use the audio file as the audio source
    r = sr.Recognizer()
    print('start of speech -recognizer')
    with sr.AudioFile(str(writing_file_path)) as source:
        audio = r.record(source)  # read the entire audio file
        print('read audio-recognizer')
        try:

            # # Reduce noise from the audio data using noisereduce
            # print('reduce noise', audio.get_raw_data())
            # reduced_noise = nr.reduce_noise(y=audio., sr=audio.sample_rate)
            # print('endreduce noise')
            # # Convert the reduced noise audio data back to AudioData
            # reduced_noise_audio = sr.AudioData(reduced_noise, audio.sample_rate, audio.sample_width)
            # print('end reduce noise file')

            # text = r.recognize_google(reduced_noise_audio)
            text = r.recognize_google(audio, language='en')
            print(text)
        except Exception as e:
            print(e)
            st.error(f"An error occurred: {str(e)}")
            text = None
            print("sorry, could not recognise")

        st.write("Transcription: " , text)
    print('end of speech -recognizer')

    return text

def image_to_base64(image):
    img_byte_array = BytesIO()
    Image.fromarray(image).save(img_byte_array, format='PNG')
    base64_str = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    return base64_str

def process_mp4(uploaded_file_path):

    audio_feature = extract_audio_features(uploaded_file_path)
    images = extract_images(uploaded_file_path)
    text = speech_to_text()

    feature = {
        'audio' : str(audio_feature),
        'text' : text,
        'images' : images
    }
    # print(feature)
    return feature

def main():
    st.title("Emotion Recognizing Engine")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    #show_table()

    if uploaded_file:
        st.video(uploaded_file)

        uploaded_file_name = uploaded_file.name
        uploaded_file_path = pathlib.Path(temp_dir.name) / uploaded_file_name

        with open(uploaded_file_path, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_file.read())

        processed = process_mp4(uploaded_file_path)

        if st.button("Send to AWS"):
            try:
                # Prepare data for the REST API request
                endpoint_url = "https://agm13flvri.execute-api.us-east-1.amazonaws.com/emotion_prediction/emotion_predictor"
                body = json.dumps({"body": processed})
                print(body)

                # Send video to AWS REST endpoint
                response = requests.post(endpoint_url, json=body)
                print(response)

                if response.status_code == 200:
                    # st.write(json.loads(response.content.decode('utf-8'))['body'])
                    st.write("Video sent to AWS endpoint successfully!")
                    final_msg = f"Predictied Emotion Category : {json.loads(response.content.decode('utf-8'))['body']}"
                    st.success(final_msg )
                else:
                    st.error("Failed to send video to AWS endpoint.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def show_table():
  # Sample data
  data = {
      'Video File': ['dia0_utt1.mp4', 'dia0_utt2.mp4'],
      'Audio File': ['sample-3s.mp3', 'sample-6s.mp3'],
      'Set of Images': [
          ['a.png', 'b.png'],
          ['c.png']
      ],
      'Text 1': ['Lorem ipsum dolor sit amet', 'Consectetur adipiscing elit'],
      'Text 2': ['Ut enim ad minim veniam', 'Quis nostrud exercitation ullamco'],
      'Text 3': ['Duis aute irure dolor in', 'Reprehenderit in voluptate velit']
  }


  # Creating a DataFrame from the sample data
  df = pd.DataFrame(data)
  st.table(df)
  # Displaying the table with images using Streamlit
  st.write("Table with Images")
  for index, row in df.iterrows():
      cols = st.columns(6)

      cols[0].video(row['Video File'])
      cols[1].audio(row['Audio File'])
      cols[2].text( row['Text 1'])
      cols[3].text( row['Text 2'])
      cols[4].text(  row['Text 3'])

      for image_filename in row['Set of Images']:
          #cols[5].image(image_filename, use_column_width=True)
          cols[5].markdown(f"<p style='text-align: center; font-size: 14px;'></p>", unsafe_allow_html=True)
          cols[5].markdown(f"<img src='{image_filename}' style='width: 100%; height: 50px; object-fit: cover;'>", unsafe_allow_html=True)

      st.write("---")  # Add a separator between rows

if __name__ == '__main__':
    main()
