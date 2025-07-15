import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import pickle

from src.vector_db import VectorDB


class TestVectorDB(unittest.TestCase):

    @patch("src.vector_db.HuggingFaceEmbeddings")
    def setUp(self, mock_embeddings):
        mock_embeddings.return_value = MagicMock()
        self.vdb = VectorDB()
        self.mock_embeddings = mock_embeddings

    def test_initialization_defaults(self):
        self.assertEqual(self.vdb.embedding_model_name, "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(self.vdb.index_name, "faiss_index")
        self.assertEqual(self.vdb.metadata_file, "metadata.pkl")
        self.mock_embeddings.assert_called_once()

    @patch("src.vector_db.FAISS")
    def test_create_from_documents_success(self, mock_faiss):
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs

        docs = [MagicMock()]
        vs = self.vdb.create_from_documents(docs)

        mock_faiss.from_documents.assert_called_once_with(
            documents=docs, embedding=self.vdb.embedding_model
        )
        self.assertEqual(vs, mock_vs)

    def test_create_from_documents_no_docs(self):
        with self.assertRaises(ValueError):
            self.vdb.create_from_documents([])

    @patch("src.vector_db.os.makedirs")
    @patch("src.vector_db.pickle.dump")
    def test_save_vectorstore(self, mock_pickle, mock_makedirs):
        mock_vs = MagicMock()
        mock_vs.docstore._dict = {"dummy": 1}
        save_dir = "test_dir"

        self.vdb.save_vectorstore(mock_vs, save_dir)

        mock_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_vs.save_local.assert_called_once_with(folder_path=save_dir, index_name=self.vdb.index_name)

        # Check metadata file written
        metadata_path = os.path.join(save_dir, self.vdb.metadata_file)
        with patch("builtins.open", mock_open()) as mocked_file:
            with open(metadata_path, "wb") as f:
                pickle.dump({}, f)
            mocked_file.assert_called_with(metadata_path, "wb")

    @patch("src.vector_db.os.path.exists", return_value=True)
    @patch("src.vector_db.FAISS.load_local")
    def test_load_vectorstore_success(self, mock_load_local, mock_exists):
        mock_vs = MagicMock()
        mock_load_local.return_value = mock_vs
        result = self.vdb.load_vectorstore("some_dir")

        mock_exists.assert_called_once_with("some_dir")
        mock_load_local.assert_called_once()
        self.assertEqual(result, mock_vs)

    @patch("src.vector_db.os.path.exists", return_value=False)
    def test_load_vectorstore_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            self.vdb.load_vectorstore("missing_dir")

    def test_get_embedding_model_info(self):
        self.vdb.embedding_model.client.get_sentence_embedding_dimension.return_value = 768
        info = self.vdb.get_embedding_model_info()
        self.assertEqual(info["model_name"], self.vdb.embedding_model_name)
        self.assertEqual(info["embedding_size"], 768)


if __name__ == "__main__":
    unittest.main()
