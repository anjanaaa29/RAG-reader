import unittest
from unittest.mock import patch, MagicMock
from src.retrieval_chain import RetrievalChain


class TestRetrievalChain(unittest.TestCase):
    def setUp(self):
        with patch("src.retrieval_chain.ChatGroq") as mock_groq:
            mock_groq.return_value = MagicMock()
            self.chain = RetrievalChain(llm_model="test-model", temperature=0.3)

    @patch("src.retrieval_chain.ChatPromptTemplate")
    def test_setup_prompts_defaults(self, mock_prompt_template):
        instance = MagicMock()
        mock_prompt_template.from_template.return_value = instance
        self.chain.setup_prompts()
        mock_prompt_template.from_template.assert_called_once()
        self.assertEqual(self.chain.base_prompt, instance)
        self.assertIn("Disclaimer", self.chain.disclaimer)

    @patch("src.retrieval_chain.create_retrieval_chain")
    @patch("src.retrieval_chain.create_stuff_documents_chain")
    def test_create_retrieval_chain(self, mock_stuff_chain, mock_retrieval_chain):
        mock_doc_chain = MagicMock()
        mock_stuff_chain.return_value = mock_doc_chain
        mock_retrieval_chain.return_value = "mock_retrieval_chain"

        mock_retriever = MagicMock()
        mock_retriever.search_kwargs = {}

        result = self.chain.create_retrieval_chain(mock_retriever, max_docs=5)
        mock_stuff_chain.assert_called_once()
        mock_retrieval_chain.assert_called_once_with(
            retriever=mock_retriever,
            combine_docs_chain=mock_doc_chain
        )
        self.assertEqual(result, "mock_retrieval_chain")
        self.assertEqual(mock_retriever.search_kwargs["k"], 5)

    def test_format_response_default(self):
        result = {
            "answer": "This is an answer",
            "context": [
                MagicMock(
                    page_content="some content " * 20,
                    metadata={"source": "src1", "author": "auth1"}
                )
            ]
        }

        formatted = self.chain.format_response(result)
        self.assertIn("answer", formatted)
        self.assertIn("sources", formatted)
        self.assertTrue(formatted["sources"][0]["content_excerpt"].endswith("..."))

    def test_format_response_with_full_context(self):
        result = {
            "answer": "This is an answer",
            "context": [MagicMock(page_content="abc", metadata={})]
        }

        formatted = self.chain.format_response(result, include_full_context=True)
        self.assertIn("context_documents", formatted)

    def test_get_retriever_valid(self):
        mock_vectorstore = MagicMock()
        mock_vectorstore.as_retriever.return_value = "retriever"

        retriever = self.chain.get_retriever(
            mock_vectorstore, k=3, search_type="mmr",
            filter_criteria={"domain": "medical"}, fetch_k=10
        )

        mock_vectorstore.as_retriever.assert_called_once()
        args, kwargs = mock_vectorstore.as_retriever.call_args
        self.assertEqual(kwargs["search_type"], "mmr")
        self.assertEqual(kwargs["search_kwargs"]["k"], 3)
        self.assertEqual(kwargs["search_kwargs"]["filter"], {"domain": "medical"})
        self.assertEqual(kwargs["search_kwargs"]["fetch_k"], 10)
        self.assertEqual(retriever, "retriever")

    def test_get_retriever_invalid_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.chain.get_retriever(MagicMock(), search_type="invalid")
        self.assertIn("Invalid search_type", str(ctx.exception))

    def test_get_retriever_missing_score_threshold(self):
        with self.assertRaises(ValueError) as ctx:
            self.chain.get_retriever(MagicMock(), search_type="similarity_score_threshold")
        self.assertIn("score_threshold is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
