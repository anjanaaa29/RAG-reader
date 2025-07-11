from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List, Dict, Any, Optional
from src.config import GenericConfig


class RetrievalChain:
    """
    RetrievalChain encapsulates a retrieval-augmented generation pipeline
    using Groq LLM + LangChain + vectorstore retriever.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        default_domain: str = "general information"
    ):
        if not GenericConfig.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in your .env file")

        self.model_name = llm_model or GenericConfig.LLM_MODEL
        self.temperature = temperature if temperature is not None else GenericConfig.LLM_TEMPERATURE

        self.llm = ChatGroq(
            temperature=self.temperature,
            model_name=self.model_name,
            api_key=GenericConfig.GROQ_API_KEY
        )

        self.default_domain = default_domain
        self.setup_prompts()

    def setup_prompts(
        self,
        domain: Optional[str] = None,
        response_rules: Optional[List[str]] = None,
        response_format: Optional[List[str]] = None,
        disclaimer: Optional[str] = None
    ):
        """
        Define the prompt template for the LLM.
        """
        domain = domain or self.default_domain

        default_rules = [
            "Use ONLY the provided context",
            "Respond at an 8th grade reading level",
            "ALWAYS cite sources",
            "Include appropriate disclaimers when needed",
            "Structure responses clearly"
        ]
        default_format = [
            "### Answer",
            "[Clear summary answer]", "",
            "### Evidence",
            "[Relevant excerpts from sources]", "",
            "### Sources",
            "[Formatted citations]"
        ]
        default_disclaimer = (
            "**Disclaimer**: This information is for educational purposes only. "
            "Consult appropriate professionals for specific advice."
        )

        # Prepare the rules and format text outside of f-string to avoid syntax errors
        rules_list = response_rules or default_rules
        format_list = response_format or default_format

        rules_text = "1. " + "\n2. ".join(rules_list)
        format_text = "\n".join(format_list)

        self.base_prompt = ChatPromptTemplate.from_template(
            f"""
            You are a {domain} specialist providing accurate information.
            Provide evidence-based responses following these rules:

            {rules_text}

            Context:
            {{context}}

            Question: {{input}}

            Required response format:
            {format_text}

            {{disclaimer}}
            """
        )

        self.disclaimer = disclaimer or default_disclaimer

    def create_retrieval_chain(
        self,
        retriever,
        max_docs: int = 7,
        custom_prompt: Optional[ChatPromptTemplate] = None,
        custom_disclaimer: Optional[str] = None
    ):
        """
        Create a LangChain retrieval chain.
        """
        prompt = custom_prompt or self.base_prompt
        disclaimer = custom_disclaimer or self.disclaimer

        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs["k"] = max_docs

        document_chain = create_stuff_documents_chain(
            self.llm,
            prompt.partial(disclaimer=disclaimer)
        )

        return create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

    def format_response(
        self,
        result: Dict[str, Any],
        metadata_fields: Optional[List[str]] = None,
        content_excerpt_length: int = 300,
        include_full_context: bool = False
    ) -> Dict[str, Any]:
        """
        Post-process LLM output: extract and format the answer & sources.
        """
        response = result.get("answer", "")

        default_fields = ["source", "author", "year", "institution", "title"]
        fields_to_extract = metadata_fields or default_fields

        sources = []
        for doc in result.get("context", []):
            source_info = {
                "content_excerpt": doc.page_content[:content_excerpt_length] +
                                   ("..." if len(doc.page_content) > content_excerpt_length else "")
            }

            for field in fields_to_extract:
                source_info[field] = doc.metadata.get(field, "")

            sources.append(source_info)

        output = {
            "answer": response,
            "sources": sources
        }

        if include_full_context:
            output["context_documents"] = result.get("context", [])

        return output

    def get_retriever(
        self,
        vectorstore,
        k: int = 7,
        search_type: str = "mmr",
        score_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict] = None,
        fetch_k: Optional[int] = None
    ):
        """
        Create a retriever with configurable search parameters.
        """
        valid_search_types = ["similarity", "mmr", "similarity_score_threshold"]
        if search_type not in valid_search_types:
            raise ValueError(f"Invalid search_type. Must be one of: {valid_search_types}")

        if search_type == "similarity_score_threshold" and score_threshold is None:
            raise ValueError("score_threshold is required for similarity_score_threshold")

        search_kwargs = {"k": k}

        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        if filter_criteria:
            search_kwargs["filter"] = filter_criteria
        if fetch_k is not None and search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k

        return vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
