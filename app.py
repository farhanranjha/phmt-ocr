from flask import Flask, request, jsonify
from Code.inferenceModel import perform_ocr_on_prescription
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/perform_ocr": {"origins": "*"}})

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    textType = request.form['textType']
    result = perform_ocr_on_prescription(image, textType)

    return jsonify({'data': result})

if __name__ == '__main__':
    app.run(debug=True)
