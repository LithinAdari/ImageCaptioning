import sys
import subprocess
import os

pdf_path = "Image_Captioning_using_Transformers (1).pdf"
out_path = "extracted_text.txt"

try:
    import pypdf
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf'])
    import pypdf

reader = pypdf.PdfReader(pdf_path)
with open(out_path, "w", encoding="utf-8") as f:
    for page in reader.pages:
        text = page.extract_text()
        if text:
            f.write(text + "\n")
print("Extraction complete.")
