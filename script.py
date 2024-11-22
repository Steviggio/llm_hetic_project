import requests
from PyPDF2 import PdfReader
from io import BytesIO

def extract_text_from_pdf_link(pdf_link):
    """Extract text directly from a PDF file accessed via a link."""
    try:
        # Step 1: Download the PDF directly into memory | Téléchargement du PDF
        response = requests.get(pdf_link)
        response.raise_for_status()  
        
        # Step 2: Read the PDF from memory | Lecture du PDF depuis la mémoire
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        # Step 3: Extract text from all pages | Extraction du text de toutes les pages
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

def save_text_to_file(text, output_file):
    """Save extracted text to a file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted text saved to {output_file}")
    except Exception as e:
        print(f"Error saving text: {e}")

# PDF link to extract text from | Lien du fichier PDF à extraire | Nom du fichier PDF lors de la sauvegarde
PDF_LINK = "https://dl.dropboxusercontent.com/1/view/8dg8ibktm7fmf77/lang_learning.pdf"
TEXT_OUTPUT_PATH = "lang_learning.txt"

# Step 1: Extract text from the PDF link | Extraction du text du fichier PDF
extracted_text = extract_text_from_pdf_link(PDF_LINK)

# Step 2: Save the extracted text to a file | Sauvegarde du text extrait dans un fichier
if extracted_text:
    save_text_to_file(extracted_text, TEXT_OUTPUT_PATH)
else:
    print("No text could be extracted from the PDF.")
