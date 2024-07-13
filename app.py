from flask import Flask, render_template, request, jsonify
from google.cloud import translate_v2 as translate
import joblib
import os

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'translate-key.json'  # Update this path

# Load the trained model
model = joblib.load('rf.pkl')  # Assuming model.pkl is in the same directory as app.py

# Initialize the translation client
translate_client = translate.Client()

def translate_text(text, target_language='en'):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

@app.route('/')
def index():
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
