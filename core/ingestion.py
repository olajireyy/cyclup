"""
ingestion.py
────────────
Document parsing helpers for CYCLUP.
Handles .txt, .docx, .pdf, and image (.jpg/.png) files.
"""

import os
import logging

logger = logging.getLogger(__name__)


def parse_txt(file_obj) -> str:
    """Read a plain .txt file and return its content as a string."""
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return file_obj.read().decode("utf-8", errors="replace")


def parse_docx(file_obj) -> str:
    """Extract all paragraph text from a .docx file using python-docx."""
    import docx

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    doc = docx.Document(file_obj)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_pdf(file_obj) -> list[dict]:
    """
    Extract text from a PDF page-by-page using PyMuPDF.

    Returns a list of dicts:
        [{"page_number": 1, "text": "..."}, ...]

    If ALL pages have zero extracted text, returns an empty list
    (indicates a scanned PDF → caller should alert the user).
    """
    import fitz  # PyMuPDF

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw_bytes = file_obj.read()
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")

        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            pages.append({"page_number": page_num + 1, "text": text})

        doc.close()
    except Exception as exc:
        logger.error("Failed to parse PDF with PyMuPDF: %s", exc)
        raise RuntimeError(f"Could not read PDF file: {exc}") from exc

    # Scanned PDF safety check: if every page is empty, return []
    if all(len(p["text"]) == 0 for p in pages):
        return []

    # Filter out truly empty pages
    return [p for p in pages if len(p["text"]) > 0]


def parse_image(file_obj) -> str:
    """
    Extract text from an image (.jpg, .png) using RapidOCR with pytesseract fallback.
    Returns the concatenated OCR text.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw_bytes = file_obj.read()
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()
        result, _ = ocr_engine(raw_bytes)
        if result:
            lines = [item[1] for item in result if item[1]]
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("RapidOCR failed or not available, trying pytesseract fallback: %s", exc)

    try:
        import io
        from PIL import Image
        import pytesseract
        image = Image.open(io.BytesIO(raw_bytes))
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:
        logger.error("All OCR methods failed: %s", exc)
        return ""




def detect_file_type(filename: str) -> str:
    """Map a filename to a source_type string."""
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        ".txt": "txt_file",
        ".docx": "docx",
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
    }
    return mapping.get(ext, "text")
