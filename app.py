from flask import Flask,render_template,url_for,json,jsonify,request
import json
from transformers import pipeline
# import googleapiclient.discovery
import os
import pandas as pd
import re
# Import the printpanda function from your module
from youtube_module.youtube_script import printpanda
from karnataka.kar_analysis import analyse_karnataka
from delhi.delhielection import analyse_delhi
from speechtotext.speechtotext import speech
import wave


app = Flask(__name__)


@app.route('/')
def home():
    img1 = url_for('static', filename='images/vote.jpg')
    img2= url_for('static', filename='images/youtube.jpg')
    img3 = url_for('static', filename='images/instagram.jpg')
    img4 = url_for('static', filename='images/facebook.jpg')
    img5 = url_for('static', filename='images/twitter.jpg')
    img6  = url_for('static', filename='images/logo1.png')
    image_path = [img1, img2, img3, img4, img5,img6]
    return render_template('home.html', image_path=image_path)

@app.route('/',methods=["POST"])
def homepost():
    b = request.form["region"]
    if b.lower()=="karnataka":
        imgstring = analyse_karnataka()
        return render_template('plot.html', img_str = imgstring)
    elif b.lower() == "delhi":
        img_string = analyse_delhi()
        return render_template('plot1.html', img_str = img_string)



@app.route('/summarize') 
def summarize(): 
    return render_template('summarize.html') 

@app.route("/summarized", methods=['POST'])
def summarized():

    speech_text = request.form['entered-text']
    print(speech_text)
    
    summarizer = pipeline("summarization")
    summary = summarizer(speech_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)
    summarized_text = summary[0]['summary_text'].strip()
    print(summary[0]['summary_text'].strip())
    return summarized_text

def extract_video_id(youtube_link):
    # Use regex to extract characters after "v="
    match = re.search(r"(?<=v=)[^&]+", youtube_link)
    return match.group() if match else None

dev = "AIzaSyC8_KDFzLWQycXSPx5TssnGqJ3B4hKydj4"

@app.route('/youtubescrap', methods=['GET', 'POST'])
def index():
    video_id = None

    if request.method == 'POST':
        youtube_link = request.form['youtube_link']
        video_id = extract_video_id(youtube_link)

        if video_id:
            # If video ID is not None, call the printpanda function
            printpanda(video_id, dev)
            return "THE data has been submitted"

    return render_template('index.html', video_id=video_id)

@app.route('/speech')
def speech_to_text():
    if request.method == 'GET':
        return render_template("speech.html")
    
@app.route('/speech', methods=['POST'])
def recognize_speech():
    if request.method == "POST":
        try:
            # Check if the POST request has the file part
            if 'wavFile' not in request.files:
                return render_template('speech.html', error='No file part')

            file = request.files['wavFile']

            # Check if the file is uploaded
            if file.filename == '':
                return render_template('speech.html', error='No selected file')

            # Check if the file has the allowed extension
            if file and file.filename.endswith('.wav'):
                recognizer = sr.Recognizer()
                audio_file = sr.AudioFile(file)
                
                with audio_file as source:
                    audio_data = recognizer.record(source)

                text = recognizer.recognize_google(audio_data)

                return render_template('speech.html', text=text)

            else:
                return render_template('speech.html', error='Invalid file type. Please upload a WAV file.')

        except Exception as e:
            return render_template('speech.html', error=f'Error: {str(e)}')

        
@app.route('/karnataka', methods=['GET', 'POST'])
def karnataka():
    if request.method == 'GET':
        imgstring = analyse_karnataka()
        return render_template('plot.html', img_str = imgstring)

@app.route('/delhi', methods=['GET', 'POST'])
def delhi():
    if request.method == 'GET':
        img_string = analyse_delhi()
        return render_template('plot1.html', img_str = img_string)
        


if __name__ == '__main__':
    app.run(debug=True)
