import os
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()


class EmbeddingModelType(str, Enum):
    DEFAULT = "default"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


class DocumentPriority(str, Enum):
    OFFICIAL = "official"
    RESEARCH = "research"
    USER_GENERATED = "user_generated"
    THIRD_PARTY = "third_party"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"


class GenericConfig:
    """Generic configuration class adaptable to multiple domains"""

    # --- Core Configuration ---
    APP_NAME = os.getenv("APP_NAME", "Knowledge Base Bot")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DOMAIN = os.getenv("DOMAIN", "general")

    # --- LLM Parameters ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    # --- Document Processing ---
    DATA_DIR = os.getenv("DATA_DIR", "data")
    MAX_DOCUMENT_LENGTH = int(os.getenv("MAX_DOCUMENT_LENGTH", 100000))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))

    # --- Embedding Models ---
    EMBEDDING_MODEL_MAP = {
        EmbeddingModelType.DEFAULT: "sentence-transformers/all-mpnet-base-v2",
        EmbeddingModelType.OPENAI: "text-embedding-3-small",
        EmbeddingModelType.HUGGINGFACE: "sentence-transformers/all-MiniLM-L6-v2",
        EmbeddingModelType.COHERE: "embed-english-v3.0"
    }
    EMBEDDING_MODEL = EmbeddingModelType(os.getenv("EMBEDDING_MODEL", "default"))

    # --- Vector Store ---
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "faiss_index")
    DOCUMENT_PRIORITY_WEIGHTS = {
        DocumentPriority.OFFICIAL: 1.5,
        DocumentPriority.RESEARCH: 1.3,
        DocumentPriority.THIRD_PARTY: 1.1,
        DocumentPriority.USER_GENERATED: 1.0
    }

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    HUGGINGFACEHUB_API_KEY: Optional[str] = os.getenv("HUGGINGFACEHUB_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")

    # --- Retrieval Parameters ---
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
    MAX_SOURCES_TO_CITE = int(os.getenv("MAX_SOURCES_TO_CITE", 3))

    # --- Safety Controls ---
    REDACT_SENSITIVE = os.getenv("REDACT_SENSITIVE", "true").lower() == "true"
    SAFETY_FILTER_LEVEL = os.getenv("SAFETY_FILTER_LEVEL", "medium")

    @classmethod
    def get_embedding_model_name(cls) -> str:
        return cls.EMBEDDING_MODEL_MAP[cls.EMBEDDING_MODEL]

    @classmethod
    def get_document_priority(cls, doc_type: str) -> float:
        try:
            return cls.DOCUMENT_PRIORITY_WEIGHTS.get(
                DocumentPriority(doc_type),
                1.0
            )
        except ValueError:
            return 1.0

    @classmethod
    def validate_config(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in your .env")


        # Validate LLM provider
        if cls.LLM_PROVIDER == LLMProvider.OPENAI:
            assert cls.OPENAI_API_KEY, "OPENAI_API_KEY required for OpenAI LLM"
        elif cls.LLM_PROVIDER == LLMProvider.GROQ:
            assert cls.GROQ_API_KEY, "GROQ_API_KEY required for Groq LLM"
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER}")

        # Validate at least one embedding key
        if cls.EMBEDDING_MODEL == EmbeddingModelType.OPENAI:
            assert cls.OPENAI_API_KEY, "OPENAI_API_KEY required for OpenAI embeddings"
        elif cls.EMBEDDING_MODEL == EmbeddingModelType.COHERE:
            assert cls.COHERE_API_KEY, "COHERE_API_KEY required for Cohere embeddings"
        # HuggingFace & DEFAULT embeddings don't need a key



# Validate at import
GenericConfig.validate_config()
