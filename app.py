from flask import Flask, render_template, request, jsonify
from model import preprocess_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    sinhala_text = request.form['sinhala_text']
    prediction = preprocess_and_predict(sinhala_text)  
    return jsonify({'prediction': prediction})  

if __name__ == '__main__':
    app.run(debug=True)
