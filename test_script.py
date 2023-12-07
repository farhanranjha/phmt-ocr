# test_script.py
import requests

url = 'http://127.0.0.1:5000/perform_ocr'
files = {'image': open('Models/04_sentence_recognition/prescription/prescrip.jpg', 'rb')}
data = {'textType': 'handwritten'}  # Use 'data' for form data

response = requests.post(url, files=files, data=data)
print(response.json())
