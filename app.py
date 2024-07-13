from flask import Flask, render_template, request, jsonify
from model import preprocess_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.form['sinhala_text']
    preprocess_and_predict(["The rabbit slept"])
    print(data)

if __name__ == '__main__':
    app.run(debug=True)
