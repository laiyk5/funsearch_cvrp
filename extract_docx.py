import sys
from docx import Document

def extract_text(path):
    doc = Document(path)
    texts = []
    for para in doc.paragraphs:
        texts.append(para.text)
    return "\n".join(texts)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <docx-path>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        print(extract_text(path))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)
