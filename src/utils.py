import re
from typing import List, Dict, Optional, Union
from langchain.schema import Document
from datetime import datetime

class ChatUtils:
    """Generic utility functions for chatbot operations across domains"""
    
    @staticmethod
    def clean_text(text: str, custom_replacements: Dict[str, str] = None) -> str:
        """
        Clean and normalize text for processing
        - Removes excessive whitespace
        - Optionally replaces custom terms/abbreviations
        - Handles special characters
        """
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply custom replacements if provided
        if custom_replacements:
            for pattern, replacement in custom_replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text
    
    @staticmethod
    def extract_key_terms(text: str, domain_terms: List[str]) -> Dict[str, int]:
        """
        Extract and count domain-specific terms from text
        Returns dictionary with term frequencies
        """
        term_counts = {}
        cleaned_text = text.lower()
        
        for term in domain_terms:
            count = len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', cleaned_text))
            if count > 0:
                term_counts[term] = count
                
        return term_counts
    
    @staticmethod
    def filter_documents_by_relevance(
        docs: List[Document], 
        query: str, 
        min_score: float = 0.65,
        score_field: str = 'score'
    ) -> List[Document]:
        """
        Filter documents based on relevance score
        Args:
            docs: List of Document objects
            query: Original search query
            min_score: Minimum relevance score threshold
            score_field: Metadata field containing relevance score
        Returns:
            Filtered list of documents
        """
        if not docs:
            return []
            
        # If scores exist in metadata, use them
        if score_field in docs[0].metadata:
            return [doc for doc in docs if doc.metadata.get(score_field, 0) >= min_score]
            
        # Fallback: Simple keyword matching if no scores available
        query_terms = set(query.lower().split())
        filtered_docs = []
        
        for doc in docs:
            content = doc.page_content.lower()
            matches = sum(1 for term in query_terms if term in content)
            if matches / max(1, len(query_terms)) >= min_score:
                filtered_docs.append(doc)
                
        return filtered_docs
    
    @staticmethod
    def generate_disclaimer(query: str, domain: str = "general") -> str:
        """
        Generate appropriate disclaimer based on query content and domain
        """
        query_lower = query.lower()
        domain = domain.lower()
        
        if domain in ["medical", "health"]:
            if any(term in query_lower for term in ['diagnos', 'symptom', 'signs of']):
                return ("**Disclaimer**: This information is not a substitute for professional advice. "
                       "Consult a qualified provider for personal guidance.")
            
            elif any(term in query_lower for term in ['treat', 'medication', 'therapy']):
                return ("**Notice**: Discuss all options with a professional. "
                       "Individual responses may vary.")
        
        elif domain in ["legal", "law"]:
            return ("**Legal Notice**: This does not constitute legal advice. "
                   "Consult an attorney for your specific situation.")
        
        elif domain in ["financial", "investment"]:
            return ("**Financial Disclaimer**: This is not financial advice. "
                   "Consult a qualified financial advisor before making decisions.")
        
        return ("**General Disclaimer**: This information is for educational purposes only. "
               "Always consult qualified professionals when needed.")
    
    @staticmethod
    def format_response(
        response: str, 
        max_length: int = 2000,
        include_disclaimer: bool = True,
        domain: str = "general"
    ) -> str:
        """
        Format LLM response for better presentation
        - Optionally includes disclaimer
        - Limits response length
        - Adds section headers when needed
        """
        # Ensure disclaimer exists if requested
        if include_disclaimer and "disclaimer" not in response.lower():
            response += "\n\n" + ChatUtils.generate_disclaimer(response, domain)
        
        # Truncate if needed (preserving complete sentences)
        if len(response) > max_length:
            last_period = response[:max_length].rfind('.')
            if last_period > 0:
                response = response[:last_period+1]
            response += " [response truncated]"
            
        return response
    
    @staticmethod
    def parse_metadata(metadata: Dict, date_fields: List[str] = None) -> Dict:
        """
        Standardize document metadata fields
        Args:
            metadata: Original metadata dictionary
            date_fields: Fields to check for date information (default: ['date', 'publication_date', 'year', 'created'])
        """
        if date_fields is None:
            date_fields = ['date', 'publication_date', 'year', 'created']
            
        standardized = {
            'source': metadata.get('source', 'Unknown'),
            'source_type': metadata.get('source_type', 'document'),
            'date': ChatUtils._parse_date(metadata, date_fields),
            'confidence': float(metadata.get('confidence', 1.0))
        }
        
        # Preserve all original metadata
        standardized.update(metadata)
        return standardized
    
    @staticmethod
    def _parse_date(metadata: Dict, date_fields: List[str]) -> Optional[str]:
        """Extract and format date from metadata"""
        for field in date_fields:
            if field in metadata:
                try:
                    date_str = str(metadata[field])
                    if re.match(r'^\d{4}$', date_str):  # Just year
                        return date_str
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):  # ISO format
                        return date_str
                    else:  # Try to parse other formats
                        dt = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                        return dt.strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    continue
        return None
    
    @staticmethod
    def log_query(
        query: str, 
        response: str, 
        sources: List[Dict],
        domain: str = "general",
        max_response_log: int = 500
    ) -> Dict:
        """
        Log queries for review (stub for actual implementation)
        Args:
            query: User's query
            response: Generated response
            sources: List of source metadata dictionaries
            domain: Domain/category of the query
            max_response_log: Maximum characters of response to log
        Returns:
            Dictionary with log entry data
        """
        # In production, this would connect to your logging system
        entry = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'query': query,
            'response_summary': response[:max_response_log],
            'sources_used': [s.get('source', 'Unknown') for s in sources],
            'source_types': list(set(s.get('source_type', 'document') for s in sources))
        }
        return entry
    
    @staticmethod
    def add_section_headers(text: str, sections: Dict[str, str]) -> str:
        """
        Add structured section headers to text
        Args:
            text: Original text
            sections: Dictionary of {section_name: section_content}
        Returns:
            Text with formatted sections
        """
        formatted = []
        for header, content in sections.items():
            formatted.append(f"## {header.upper()} ##\n{content}\n")
        return "\n".join(formatted)