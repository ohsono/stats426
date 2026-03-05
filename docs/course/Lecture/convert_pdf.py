import pypdf
import argparse
import os

def extract_text(pdf_path, output_path):
    print(f"Extracting text from {pdf_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    argsparser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    argsparser.add_argument("--pdf_path", help="Path to the PDF file to extract text from.")
    argsparser.add_argument("--output_path", help="Path to save the extracted text file.", default="extracted_text.txt")    
    args = argsparser.parse_args()
    pdf_path = args.pdf_path    
    output_path = args.output_path
    extract_text(pdf_path, output_path)
    log_file = os.path.join(os.path.dirname(output_path), "extraction_log.txt")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"Extracted text from {pdf_path} to {output_path}\n") 