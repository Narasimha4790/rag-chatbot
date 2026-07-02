import os
import csv
from pypdf import PdfReader
from docx import Document

def parse_txt(file_path: str) -> str:
    """Reads a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_pdf(file_path: str) -> str:
    """Extracts text page by page from a PDF file."""
    reader = PdfReader(file_path)
    text_content = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
    return "\n".join(text_content)

def parse_docx(file_path: str) -> str:
    """Extracts text paragraph by paragraph from a Word document."""
    doc = Document(file_path)
    text_content = []
    for paragraph in doc.paragraphs:
        if paragraph.text:
            text_content.append(paragraph.text)
    return "\n".join(text_content)

def parse_csv(file_path: str) -> str:
    """Parses a CSV file and converts each row into a readable text sentence."""
    text_content = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return ""
        
        headers = [h.strip() for h in headers]
        for row_idx, row in enumerate(reader, start=1):
            row_items = []
            for col_idx, cell in enumerate(row):
                header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
                row_items.append(f"{header}: {cell.strip()}")
            text_content.append(f"Row {row_idx}: " + ", ".join(row_items))
            
    return "\n".join(text_content)

def extract_text(file_path: str) -> str:
    """Dispatches parsing based on file extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".csv":
        return parse_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
