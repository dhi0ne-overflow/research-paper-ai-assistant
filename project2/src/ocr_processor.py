import pytesseract
from pdf2image import convert_from_path

# Set this path manually if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_text = ""

    for img in images:
        ocr_text += pytesseract.image_to_string(img)

    return ocr_text