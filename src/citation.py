from typing import Dict, List, Optional
import re

class GenericCitationFormatter:
    """Formats citations for various document types according to academic standards"""
    
    @staticmethod
    def format_citation(doc_metadata: Dict) -> str:
        """
        Formats citation based on document source type
        Args:
            doc_metadata: Metadata from Document object
        Returns:
            Formatted citation string
        """
        source_type = doc_metadata.get("source_type", "").lower()
        
        if source_type == "official_guideline":
            return GenericCitationFormatter._format_guideline_citation(doc_metadata)
        elif source_type == "research_article":
            return GenericCitationFormatter._format_research_citation(doc_metadata)
        elif source_type == "educational_material":
            return GenericCitationFormatter._format_educational_citation(doc_metadata)
        elif source_type == "web_page":
            return GenericCitationFormatter._format_web_citation(doc_metadata)
        else:
            return GenericCitationFormatter._format_default_citation(doc_metadata)
    
    @staticmethod
    def _format_guideline_citation(metadata: Dict) -> str:
        """Format citations for official guidelines and reports"""
        organization = metadata.get("organization", "Official Source")
        year = GenericCitationFormatter._extract_year(metadata)
        title = metadata.get("title", "")
        version = metadata.get("version", "")
        report_number = metadata.get("report_number", "")
        
        citation = f"[{organization}"
        if title:
            short_title = GenericCitationFormatter._shorten_title(title)
            citation += f" {short_title}"
        if report_number:
            citation += f" {report_number}"
        if version:
            citation += f" v{version}"
        if year:
            citation += f" {year}"
        citation += "]"
        
        return citation
    
    @staticmethod
    def _format_research_citation(metadata: Dict) -> str:
        """Format citations for research articles"""
        authors = metadata.get("authors", "")
        journal = metadata.get("journal", "")
        year = GenericCitationFormatter._extract_year(metadata)
        title = metadata.get("title", "")
        doi = metadata.get("doi", "")
        url = metadata.get("url", "")
        
        # Format author list
        author_list = authors.split(",") if authors else []
        if len(author_list) > 2:
            authors_str = f"{author_list[0]} et al."
        elif author_list:
            authors_str = " & ".join(author_list)
        else:
            authors_str = "Anonymous"
        
        # Build citation
        citation = f"[{authors_str}"
        if title:
            citation += f" '{GenericCitationFormatter._shorten_title(title, 6)}'"
        if journal:
            citation += f", {journal}"
        if year:
            citation += f" {year}"
        if doi:
            citation += f" DOI:{doi}"
        elif url:
            citation += f" URL:{GenericCitationFormatter._shorten_url(url)}"
        citation += "]"
        
        return citation
    
    @staticmethod
    def _format_educational_citation(metadata: Dict) -> str:
        """Format citations for educational materials"""
        publisher = metadata.get("publisher", "Educational Material")
        year = GenericCitationFormatter._extract_year(metadata)
        title = metadata.get("title", "")
        
        citation = f"[{publisher}"
        if title:
            citation += f" {GenericCitationFormatter._shorten_title(title)}"
        if year:
            citation += f" {year}"
        citation += "]"
        
        return citation
    
    @staticmethod
    def _format_web_citation(metadata: Dict) -> str:
        """Format citations for web pages"""
        site_name = metadata.get("site_name", "Website")
        page_title = metadata.get("page_title", "")
        url = metadata.get("url", "")
        publish_date = GenericCitationFormatter._extract_year(metadata)
        access_date = metadata.get("access_date", "")
        
        citation = f"[{site_name}"
        if page_title:
            citation += f" '{GenericCitationFormatter._shorten_title(page_title)}'"
        if publish_date:
            citation += f" {publish_date}"
        if url:
            citation += f" URL:{GenericCitationFormatter._shorten_url(url)}"
        if access_date:
            citation += f" (accessed {access_date})"
        citation += "]"
        
        return citation
    
    @staticmethod
    def _format_default_citation(metadata: Dict) -> str:
        """Fallback citation format"""
        source = metadata.get("source", "Source")
        title = metadata.get("title", "")
        year = GenericCitationFormatter._extract_year(metadata)
        
        citation = f"[{source}"
        if title:
            citation += f" {GenericCitationFormatter._shorten_title(title)}"
        if year:
            citation += f" {year}"
        citation += "]"
        
        return citation
    
    @staticmethod
    def _extract_year(metadata: Dict) -> str:
        """Extract year from various metadata fields"""
        for field in ["year", "publication_date", "date", "publish_date"]:
            if field in metadata:
                date_str = str(metadata[field])
                if match := re.search(r"\b(19|20)\d{2}\b", date_str):
                    return match.group()
        return ""
    
    @staticmethod
    def _shorten_title(title: str, max_words: int = 5) -> str:
        """Shorten long document titles for citations"""
        words = title.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return title
    
    @staticmethod
    def _shorten_url(url: str, max_length: int = 30) -> str:
        """Shorten long URLs for citations"""
        if len(url) > max_length:
            return url[:max_length] + "..."
        return url
    
    @staticmethod
    def generate_source_footnotes(sources: List[Dict]) -> str:
        """
        Generate formatted footnotes for multiple sources
        Args:
            sources: List of source metadata dictionaries
        Returns:
            Formatted footnote stringa
        """
        if not sources:
            return ""
            
        footnotes = "\n\n**References:**\n"
        for i, source in enumerate(sources, start=1):
            citation = GenericCitationFormatter.format_citation(source)
            footnotes += f"{i}. {citation}\n"
        
        return footnotes