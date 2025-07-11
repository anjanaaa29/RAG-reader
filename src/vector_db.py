from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional, Dict, Any
from langchain.schema import Document
import os
import pickle

class VectorDB:
    """
    A generic vector database manager using FAISS for document storage and retrieval.
    """
    
    def __init__(self, embedding_model: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize the vector database with specified embeddings.
        
        Args:
            embedding_model: Name of HuggingFace sentence transformer model
            index_name: Custom name for the FAISS index (optional)
        """
        self.embedding_model_name = embedding_model or "sentence-transformers/all-mpnet-base-v2"
        self.embedding_model = self._get_embeddings()
        self.index_name = index_name or "faiss_index"
        self.metadata_file = "metadata.pkl"

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings model"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_from_documents(self, 
                            documents: List[Document], 
                            save_dir: Optional[str] = None) -> FAISS:
        """
        Create FAISS vectorstore from documents.
        
        Args:
            documents: List of Document objects to index
            save_dir: Directory to save the vectorstore (optional)
            
        Returns:
            FAISS vectorstore instance
        """
        if not documents:
            raise ValueError("No documents provided for vectorstore creation")

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        if save_dir:
            self.save_vectorstore(vectorstore, save_dir)

        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, save_dir: str) -> None:
        """Save vectorstore to disk"""
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(
            folder_path=save_dir,
            index_name=self.index_name
        )
        
        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "document_count": len(vectorstore.docstore._dict),
            "index_name": self.index_name
        }
        with open(os.path.join(save_dir, self.metadata_file), "wb") as f:
            pickle.dump(metadata, f)

    def load_vectorstore(self, load_dir: str) -> FAISS:
        """Load vectorstore from disk"""
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Vectorstore directory not found: {load_dir}")
        
        return FAISS.load_local(
            folder_path=load_dir,
            embeddings=self.embedding_model,
            index_name=self.index_name,
            allow_dangerous_deserialization=True
        )




    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model_name": self.embedding_model_name,
            "embedding_size": self.embedding_model.client.get_sentence_embedding_dimension()
        }