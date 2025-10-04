import PyPDF2
import sys

pdf_path = 'Εφαρμογές τεχνητής νοημοσύνης και ChatGPT σε κρίσιμους τομείς.pdf'

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        print(f"Number of pages: {len(reader.pages)}\n")
        print("="*80)
        
        for i, page in enumerate(reader.pages):
            print(f"\n--- PAGE {i+1} ---\n")
            text = page.extract_text()
            print(text)
            print("\n" + "="*80)
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
