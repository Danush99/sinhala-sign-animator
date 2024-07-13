from flask import Flask, render_template, request, jsonify
from google.cloud import translate_v2 as translate
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import os
import fasttext
from model import preprocess_and_predict

app = Flask(__name__)


@app.route('/')
def index():
    preprocess_and_predict("The rabbit slept.")
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.form['sinhala_text']
    try:
        # Translate the Sinhala text
        english_text = translate_text(data)
        
        # Assume you have a preprocessing function for the translated text
        processed_text = preprocess(english_text)
        
        # Predict tags
        # prediction = model.predict([processed_text])
        
        return jsonify({'translatedText': english_text, 'prediction': '["Shani ❤️"]'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess(text):
    # Implement your preprocessing here if needed
    return text

if __name__ == '__main__':
    app.run(debug=True)
