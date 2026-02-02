"""
Resume Text Extractor
Extracts text from various resume formats (PDF, DOCX, TXT)
"""

import os
from typing import Optional
import PyPDF2
from docx import Document


class ResumeExtractor:
    """Extract text from resume files in various formats"""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """
        Extract text from TXT file
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from resume file (auto-detects format)
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Extract based on file type
        if ext == '.pdf':
            return ResumeExtractor.extract_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return ResumeExtractor.extract_from_docx(file_path)
        elif ext == '.txt':
            return ResumeExtractor.extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: PDF, DOCX, TXT")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python resume_extractor.py <resume_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        extractor = ResumeExtractor()
        text = extractor.extract_text(file_path)
        
        print("="*60)
        print("EXTRACTED RESUME TEXT")
        print("="*60)
        print(text)
        print("="*60)
        print(f"\nTotal characters: {len(text)}")
        print(f"Total words: {len(text.split())}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()