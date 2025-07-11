import os
import json
from typing import List, Dict, Union
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader,
    TextLoader,
    CSVLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

class DocumentLoader:
    """
    Generic document loader supporting multiple file types with metadata preservation
    Supports PDFs, Word docs, JSON, TXT, CSV files
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize with configurable text splitting parameters
        
        Args:
            chunk_size: Size of each text chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n• ", "\n", ". ", " ", ""],
            length_function=len
        )
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF documents with metadata extraction
        """
        try:
            # Extract metadata first using PyMuPDF
            pdf_metadata = {}
            with fitz.open(file_path) as doc:
                pdf_metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "publication_date": doc.metadata.get("creationDate", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "source_type": "pdf",
                    "file_path": file_path
                }
            
            # Load content with PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Enhance metadata
            for page in pages:
                page.metadata.update(pdf_metadata)
                
                # Extract section headers if present
                if "page" in page.metadata:
                    page.metadata["section"] = self._extract_section_header(page.page_content)
            
            return self.text_splitter.split_documents(pages)
            
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            return []

    def load_json(self, json_path: str, jq_schema: str = None) -> List[Document]:
        """
        Load JSON documents with customizable schema
        
        Args:
            json_path: Path to JSON file
            jq_schema: Optional jq schema for parsing JSON (defaults to basic structure)
        """
        try:
            if jq_schema is None:
                jq_schema = """
                .[] | {
                    text: .text,
                    metadata: {
                        title: .title,
                        author: .author,
                        date: .date,
                        source: .source,
                        file_path: "%s"
                    }
                }
                """ % json_path
            
            loader = JSONLoader(
                file_path=json_path,
                jq_schema=jq_schema,
                text_content=False
            )
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
            
        except Exception as e:
            print(f"Error loading JSON {json_path}: {str(e)}")
            return []

    def load_word_document(self, file_path: str) -> List[Document]:
        """
        Load Word documents (both .doc and .docx)
        """
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "word_document",
                    "file_path": file_path
                })
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading Word document {file_path}: {str(e)}")
            return []

    def load_text_file(self, file_path: str) -> List[Document]:
        """
        Load plain text files
        """
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "text_file",
                    "file_path": file_path
                })
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading text file {file_path}: {str(e)}")
            return []

    def load_csv(self, file_path: str) -> List[Document]:
        """
        Load CSV files
        """
        try:
            loader = CSVLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "csv_file",
                    "file_path": file_path
                })
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {str(e)}")
            return []

    def _extract_section_header(self, text: str) -> str:
        """
        Extract potential section headers from document text
        """
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if (line.isupper() or 
                line.endswith(':') or 
                any(word in line.lower() for word in ['section', 'chapter', 'heading'])):
                return line
        return ""

    def load_document(self, file_path: str, **kwargs) -> List[Document]:
        """
        Auto-detect and load document based on file type
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments for specific loaders
                      (e.g., jq_schema for JSON files)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.load_pdf(file_path)
        elif file_ext == '.json':
            jq_schema = kwargs.get('jq_schema')
            return self.load_json(file_path, jq_schema)
        elif file_ext in ('.doc', '.docx'):
            return self.load_word_document(file_path)
        elif file_ext == '.txt':
            return self.load_text_file(file_path)
        elif file_ext == '.csv':
            return self.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")