from flask import Flask, request, jsonify
from Code.inferenceModel import perform_ocr_on_prescription

app = Flask(__name__)

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    result = perform_ocr_on_prescription(image)

    return jsonify({'data': result})

if __name__ == '__main__':
    app.run(debug=True)
