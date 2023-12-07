import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer
import typing
import shutil
import easyocr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class PrescribedMedicine:
    def __init__(self, medicine_name=None, dosage=None, quantity=None, frequency=None):
        self.medicine_name = medicine_name
        self.dosage = dosage
        self.quantity = quantity
        self.frequency = frequency
    def to_dict(self):
        return {
            'medicine_name': self.medicine_name,
            'dosage': self.dosage,
            'quantity': self.quantity,
            'frequency': self.frequency
        }

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def increase_image_size(input_path, output_path, scale_factor=2):
    try:
        with Image.open(input_path) as img:
            original_width, original_height = img.size
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor
            resized_img = img.resize((new_width, new_height))
            resized_img.save(output_path)
    except Exception as e:
        print(f"Error resizing image: {str(e)}")

def save_non_empty_cells(table_image, output_folder="/Users/apple/Desktop/medifyme/medifyme-ocr/Models/04_sentence_recognition/cells/", boundary_margin=2):
    os.makedirs(output_folder, exist_ok=True)

    num_rows = 6
    num_columns = 4

    cell_width = table_image.width // num_columns
    cell_height = table_image.height // num_rows

    table_data = pd.DataFrame(index=range(num_rows), columns=range(num_columns))

    for i in range(1, num_rows):
        for j in range(num_columns):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height

            x1 += boundary_margin
            y1 += boundary_margin
            x2 -= boundary_margin
            y2 -= boundary_margin

            cell_image_with_border = table_image.crop((x1, y1, x2, y2))
            border_size = 10
            cell_image = cell_image_with_border.crop((border_size, border_size, cell_image_with_border.width - border_size, cell_image_with_border.height - border_size))

            cell_array = np.array(cell_image)
            cell_intensity = np.mean(cell_array)
            if cell_intensity > 200:
                cell_image.save(f"{output_folder}cell_{i}_{j}.png", dpi=(300, 300))
                increase_image_size(f"{output_folder}cell_{i}_{j}.png",f"{output_folder}cell_{i}_{j}.png", scale_factor=2)
                table_data.at[i, j] = f"Cell {i}_{j}"

def process_prescribed_medicines(model, images_folder, textType):
    prescribed_medicines = []
    current_medicine = None

    image_filenames = sorted(os.listdir(images_folder), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))
    reader = easyocr.Reader(['en'])
    for filename in image_filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)

            if image is None or image.size == 0:
                print(f"Error: Could not open or read the image at {image_path}")
            else:
                if filename.endswith('0.png') and textType == 'handwritten':
                    prediction_text = model.predict(image) if model.predict(image) else "n/a"
                else:
                    prediction_text = reader.readtext(image)[0][1] if reader.readtext(image) else "n/a"

                if prediction_text:
                    if current_medicine is None:
                        current_medicine = PrescribedMedicine()

                    if current_medicine.medicine_name is None:
                        current_medicine.medicine_name = prediction_text
                    elif current_medicine.dosage is None:
                        current_medicine.dosage = prediction_text
                    elif current_medicine.quantity is None:
                        current_medicine.quantity = prediction_text
                    elif current_medicine.frequency is None:
                        current_medicine.frequency = prediction_text
                else:
                    break

                if current_medicine and current_medicine.frequency:
                    prescribed_medicines.append(current_medicine)
                    current_medicine = None

    if current_medicine:
        prescribed_medicines.append(current_medicine)

    return prescribed_medicines

def perform_ocr_on_prescription(image, textType):
    nparr = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, largest_contours, -1, (255), thickness=cv2.FILLED)
    table_highlighted = cv2.bitwise_and(image, image, mask=mask)
    cv2.drawContours(image, largest_contours, -1, (0, 255, 0), thickness=2)
    gray_highlighted = cv2.cvtColor(table_highlighted, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_highlighted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Use only the largest contour
    mask = np.zeros_like(gray, dtype=np.uint8)  # Ensure the mask is in uint8 format
    cv2.drawContours(mask, largest_contours, -1, (255), thickness=cv2.FILLED)  # Use (255) for single-channel mask
    x, y, w, h = cv2.boundingRect(mask)
    table_roi = image[y:y + h, x:x + w]

    cv2.imwrite('/Users/apple/Desktop/medifyme/medifyme-ocr/Models/04_sentence_recognition/prescription/table_roi.png', table_roi)
    configs = BaseModelConfigs.load("/Users/apple/Desktop/medifyme/medifyme-ocr/Models/04_sentence_recognition/202301131202/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    
    table_image_path = "/Users/apple/Desktop/medifyme/medifyme-ocr/Models/04_sentence_recognition/prescription/table_roi.png"
    table_image = Image.open(table_image_path)
    save_non_empty_cells(table_image)

    images_folder = "/Users/apple/Desktop/medifyme/medifyme-ocr/Models/04_sentence_recognition/cells/"
    result = process_prescribed_medicines(model, images_folder, textType)
    shutil.rmtree(images_folder)
    
    # for idx, medicine in enumerate(result, 0):
    #     print(f"Prescribed Medicine {idx}:")
    #     print(f"Medicine Name: {medicine.medicine_name}")
    #     print(f"Dosage: {medicine.dosage}")
    #     print(f"Quantity: {medicine.quantity}")
    #     print(f"Frequency: {medicine.frequency}")
    #     print()
    
    result_dict_list = [medicine.to_dict() for medicine in result]    
    return result_dict_list