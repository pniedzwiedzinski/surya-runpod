from io import BytesIO
import time
from collections import defaultdict
from typing import List, Tuple

import requests
from PIL import Image

# Import required functions/classes from surya.
from surya.detection import DetectionPredictor
from surya.recognition.languages import replace_lang_with_code
from surya.recognition import RecognitionPredictor

# Global configuration
DEBUG_MODE = True

def read_image(file_path: str) -> Image.Image:
    """
    Convert an uploaded file to a PIL Image.
    """
    if isinstance(file_path, str):
        if file_path.startswith("http"):
            response = requests.get(file_path)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError("Invalid file path provided.")
    else:
        raise ValueError("File path must be a string.")


def load_images(files: List[str]) -> Tuple[List[Image.Image], List[str]]:
    """
    Load images from uploaded files and return a list of PIL Images along with their names.
    """
    images = []
    names = []
    for idx, file in enumerate(files):
        img = read_image(file)
        images.append(img)
        names.append(f"image_{idx}")
    return images, names

def get_languages(num_images: int) -> List[List[str]]:
    """
    Return a list of languages for each image. Here the language is hardcoded to 'pl'.
    For every image, the OCR language list will contain only ['pl'].
    """
    # Hardcode the language to Polish ("pl") and update it to ISO code if needed
    lang = "pl"
    replace_lang_with_code([lang])
    return [[lang] for _ in range(num_images)]

def run_ocr(images: List[Image.Image], image_langs: List[List[str]]) -> List:
    """
    Run the OCR process on the provided images.
    """
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    start_time = time.time()
    predictions = rec_predictor(
        images,
        image_langs,
        det_predictor=det_predictor,
        highres_images=images
    )
    processing_time = time.time() - start_time

    if DEBUG_MODE:
        print(f"OCR processing took {processing_time:.2f} seconds")
        # If there is any text, print maximum characters in a text line.
        max_chars = max((len(line.text) for pred in predictions for line in pred.text_lines), default=0)
        print(f"Max characters in any text line: {max_chars}")

    return predictions, processing_time

def prepare_response(names: List[str], predictions, processing_time: float) -> dict:
    """
    Prepare the JSON response from OCR predictions.
    """
    out_preds = defaultdict(list)
    for name, pred in zip(names, predictions):
        dump = pred.model_dump()  # Serialize prediction data.
        dump["page"] = len(out_preds[name]) + 1
        out_preds[name].append(dump)

    return {
        "processing_time": processing_time,
        "results": out_preds
    }


def ocr_text(image_paths: List[str]) -> dict:

    # Load images and filenames.
    loaded_images, names = load_images(image_paths)
    
    # Get languages for each image; hardcoded to "pl"
    image_langs = get_languages(len(loaded_images))
    
    # Perform OCR processing.
    predictions, processing_time = run_ocr(loaded_images, image_langs)
    
    # Prepare and return JSON response.
    response = prepare_response(names, predictions, processing_time)
    return response
