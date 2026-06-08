import pdfplumber
import re


class PDFExtractor:
    def extract(self, file_path: str) -> str:
        """Extract and clean text from a PDF file."""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    pages.append(text)

        if not pages:
            return ""

        raw = "\n".join(pages)
        return self._clean(raw)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" \n", "\n", text)
        return text.strip()
