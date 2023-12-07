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
        # Open the image
        with Image.open(input_path) as img:
            # Get the original width and height
            original_width, original_height = img.size

            # Calculate the new width and height
            new_width = original_width * scale_factor
            new_height = original_height * scale_factor

            # Resize the image
            resized_img = img.resize((new_width, new_height))

            # Save the resized image
            resized_img.save(output_path)
            print(f"Image resized successfully: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error resizing image: {str(e)}")

# Function to process the table and save non-empty cells as images
def save_non_empty_cells(table_image, output_folder="/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/cells/", boundary_margin=2):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the number of rows and columns in the table
    num_rows = 6
    num_columns = 4

    # Get the width and height of each cell
    cell_width = table_image.width // num_columns
    cell_height = table_image.height // num_rows

    # Initialize pandas DataFrame to store table data
    table_data = pd.DataFrame(index=range(num_rows), columns=range(num_columns))

    # Iterate through each cell in the table
    for i in range(1, num_rows):
        for j in range(num_columns):
            # Define the coordinates of the current cell with a boundary margin
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height

            # Apply boundary margin
            x1 += boundary_margin
            y1 += boundary_margin
            x2 -= boundary_margin
            y2 -= boundary_margin

            # Crop the cell from the table image
            cell_image_with_border = table_image.crop((x1, y1, x2, y2))
            border_size = 2  # Adjust this value based on your border size
            cell_image = cell_image_with_border.crop((border_size, border_size, cell_image_with_border.width - border_size, cell_image_with_border.height - border_size))

            # Check if the cell is non-empty
            # For demonstration purposes, let's consider a simple check based on pixel intensity
            cell_array = np.array(cell_image)
            cell_intensity = np.mean(cell_array)
            if cell_intensity > 200:  # Adjust the threshold based on your image characteristics
                # Save the non-empty cell as an image with higher resolution
                cell_image.save(f"{output_folder}cell_{i}_{j}.png", dpi=(300, 300))
                increase_image_size(f"{output_folder}cell_{i}_{j}.png",f"{output_folder}cell_{i}_{j}.png", scale_factor=2)
                # Store some placeholder data in the DataFrame
                table_data.at[i, j] = f"Cell {i}_{j}"

    # Print the table data
    print("Table Data:")
    print(table_data)

def process_prescribed_medicines(model, images_folder):
    prescribed_medicines = []
    current_medicine = None

    # Get a list of image filenames and sort them based on the cell number and index
    image_filenames = sorted(os.listdir(images_folder), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))
    reader = easyocr.Reader(['en'])
    # Iterate over sorted images
    for filename in image_filenames:
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full path to the image
            image_path = os.path.join(images_folder, filename)

            # Read the image
            image = cv2.imread(image_path)

            # Check if the image is successfully loaded
            if image is None or image.size == 0:
                print(f"Error: Could not open or read the image at {image_path}")
            else:
                if filename.endswith('0.png'):
                # Use model for prediction
                    prediction_text = model.predict(image)
                else:
                # Use easyocr for prediction
                    prediction_text = reader.readtext(image)
                # Make a prediction
                prediction_text = model.predict(image)
                # If predicted text is not empty, update or create a new medicine object
                if prediction_text:
                    if current_medicine is None:
                        current_medicine = PrescribedMedicine()

                    # Update attributes of the current medicine based on the order
                    if current_medicine.medicine_name is None:
                        current_medicine.medicine_name = prediction_text
                    elif current_medicine.dosage is None:
                        current_medicine.dosage = prediction_text
                    elif current_medicine.quantity is None:
                        current_medicine.quantity = prediction_text
                    elif current_medicine.frequency is None:
                        current_medicine.frequency = prediction_text
                else:
                    # If predicted text is empty, stop creating new medicine objects
                    break

                print(f"Image: {image_path}, Prediction: {prediction_text}")

                # Check if the current medicine is fully processed
                if current_medicine and current_medicine.frequency:
                    prescribed_medicines.append(current_medicine)
                    current_medicine = None

    # If there's a partially processed medicine, add it to the list
    if current_medicine:
        prescribed_medicines.append(current_medicine)

    return prescribed_medicines

def perform_ocr_on_prescription(image):
    # Read the image from the file object
    # nparr = np.frombuffer(image.read(), np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to highlight the table boundaries
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and find the largest contour(s) which represents the table
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 

    # Initialize an empty mask to draw the table boundaries
    mask = np.zeros_like(gray, dtype=np.uint8)  # Ensure the mask is in uint8 format

    # Draw the table boundaries on the mask
    cv2.drawContours(mask, largest_contours, -1, (255), thickness=cv2.FILLED)  # Use (255) for single-channel mask

    # Bitwise AND the original image with the mask to highlight the table
    table_highlighted = cv2.bitwise_and(image, image, mask=mask)

    # Visualize the detected contours and highlighted table
    cv2.drawContours(image, largest_contours, -1, (0, 255, 0), thickness=2)  # Draw green contours on the original image

    # Convert the image to grayscale
    gray_highlighted = cv2.cvtColor(table_highlighted, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    contours, _ = cv2.findContours(gray_highlighted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and find the largest contour(s) which represent the table
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Use only the largest contour

    # Initialize an empty mask to draw the table boundaries
    mask = np.zeros_like(gray, dtype=np.uint8)  # Ensure the mask is in uint8 format

    # Draw the table boundaries on the mask
    cv2.drawContours(mask, largest_contours, -1, (255), thickness=cv2.FILLED)  # Use (255) for single-channel mask

    # Find the bounding box of the masked area (the table)
    x, y, w, h = cv2.boundingRect(mask)

    # Crop the table region from the original image
    table_roi = image[y:y + h, x:x + w]

    cv2.imwrite('/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/prescription/table_roi.png', table_roi)
    configs = BaseModelConfigs.load("/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/202301131202/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    
    table_image_path = "/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/prescription/table_roi.png"
    table_image = Image.open(table_image_path)
    save_non_empty_cells(table_image)

    images_folder = "/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/cells/"
    result = process_prescribed_medicines(model, images_folder)
    shutil.rmtree(images_folder)
    # Print the list of prescribed medicines
    for idx, medicine in enumerate(result, 0):
        print(f"Prescribed Medicine {idx}:")
        print(f"Medicine Name: {medicine.medicine_name}")
        print(f"Dosage: {medicine.dosage}")
        print(f"Quantity: {medicine.quantity}")
        print(f"Frequency: {medicine.frequency}")
        print()
    result_dict_list = [medicine.to_dict() for medicine in result]
    return result_dict_list


image = cv2.imread('/Users/apple/Desktop/INFER 2/Models/04_sentence_recognition/prescription/prescrip.jpg')
perform_ocr_on_prescription(image)