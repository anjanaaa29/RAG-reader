import unittest
from unittest.mock import patch, MagicMock
from document_loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader(chunk_size=500, chunk_overlap=50)

    @patch("document_loader.PyPDFLoader")
    @patch("document_loader.fitz.open")
    def test_load_pdf(self, mock_fitz_open, mock_pdf_loader):
        # mock metadata from fitz
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test PDF",
            "author": "Author",
            "creationDate": "2024-01-01",
            "keywords": "test"
        }
        mock_fitz_open.return_value.__enter__.return_value = mock_doc
        
        # mock pages from PyPDFLoader
        mock_page = MagicMock()
        mock_page.page_content = "SECTION 1\nContent"
        mock_page.metadata = {"page": 1}
        mock_pdf_loader.return_value.load.return_value = [mock_page]

        result = self.loader.load_pdf("dummy.pdf")
        self.assertTrue(result)
        mock_pdf_loader.assert_called_once()
        mock_fitz_open.assert_called_once()
        self.assertIn("title", mock_page.metadata)
        self.assertEqual(mock_page.metadata["section"], "SECTION 1")

    @patch("document_loader.JSONLoader")
    def test_load_json(self, mock_json_loader):
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_json_loader.return_value.load.return_value = [mock_doc]

        result = self.loader.load_json("dummy.json")
        self.assertTrue(result)
        mock_json_loader.assert_called_once()

    @patch("document_loader.UnstructuredWordDocumentLoader")
    def test_load_word_document(self, mock_word_loader):
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_word_loader.return_value.load.return_value = [mock_doc]

        result = self.loader.load_word_document("dummy.docx")
        self.assertTrue(result)
        mock_word_loader.assert_called_once()
        self.assertIn("source_type", mock_doc.metadata)

    @patch("document_loader.TextLoader")
    def test_load_text_file(self, mock_text_loader):
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_text_loader.return_value.load.return_value = [mock_doc]

        result = self.loader.load_text_file("dummy.txt")
        self.assertTrue(result)
        mock_text_loader.assert_called_once()
        self.assertIn("source_type", mock_doc.metadata)

    @patch("document_loader.CSVLoader")
    def test_load_csv(self, mock_csv_loader):
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_csv_loader.return_value.load.return_value = [mock_doc]

        result = self.loader.load_csv("dummy.csv")
        self.assertTrue(result)
        mock_csv_loader.assert_called_once()
        self.assertIn("source_type", mock_doc.metadata)

    def test_extract_section_header(self):
        text = "SECTION 1: Introduction\nMore text\nEven more"
        header = self.loader._extract_section_header(text)
        self.assertEqual(header, "SECTION 1: Introduction")

        no_header = self.loader._extract_section_header("just some text\nmore text")
        self.assertEqual(no_header, "")

    @patch.object(DocumentLoader, "load_pdf")
    def test_load_document_pdf(self, mock_load_pdf):
        mock_load_pdf.return_value = ["pdf"]
        result = self.loader.load_document("file.pdf")
        self.assertEqual(result, ["pdf"])
        mock_load_pdf.assert_called_once()

    @patch.object(DocumentLoader, "load_json")
    def test_load_document_json(self, mock_load_json):
        mock_load_json.return_value = ["json"]
        result = self.loader.load_document("file.json")
        self.assertEqual(result, ["json"])
        mock_load_json.assert_called_once()

    @patch.object(DocumentLoader, "load_word_document")
    def test_load_document_word(self, mock_load_word):
        mock_load_word.return_value = ["word"]
        result = self.loader.load_document("file.docx")
        self.assertEqual(result, ["word"])
        mock_load_word.assert_called_once()

    @patch.object(DocumentLoader, "load_text_file")
    def test_load_document_txt(self, mock_load_txt):
        mock_load_txt.return_value = ["txt"]
        result = self.loader.load_document("file.txt")
        self.assertEqual(result, ["txt"])
        mock_load_txt.assert_called_once()

    @patch.object(DocumentLoader, "load_csv")
    def test_load_document_csv(self, mock_load_csv):
        mock_load_csv.return_value = ["csv"]
        result = self.loader.load_document("file.csv")
        self.assertEqual(result, ["csv"])
        mock_load_csv.assert_called_once()

    def test_load_document_unsupported(self):
        with self.assertRaises(ValueError) as ctx:
            self.loader.load_document("file.xyz")
        self.assertIn("Unsupported file type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
