from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from typing import Dict, List, Optional

class PromptFactory:
    """Generic factory class for creating domain-specific prompt templates"""
    
    @staticmethod
    def get_base_prompt(
        domain: str = "information",
        requirements: Optional[List[str]] = None,
        structure: Optional[List[str]] = None
    ) -> ChatPromptTemplate:
        """Core prompt template that can be customized for different domains"""
        default_requirements = [
            "Evidence-based (only use provided context)",
            "At 8th grade reading level",
            "Properly cited",
            "Contain appropriate disclaimers"
        ]
        
        default_structure = [
            "1. Clear answer",
            "2. Supporting evidence (quoted)",
            "3. Source citations",
            "4. Relevant disclaimers"
        ]
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are a {domain} specialist providing accurate information.
            Your responses must be:
            - {'\n- '.join(requirements or default_requirements)}
            
            Follow this structure:
            {'\n'.join(structure or default_structure)}
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ])
    
    @staticmethod
    def get_specialized_prompt(
        warning: Optional[str] = None,
        requirements: List[str] = None,
        example_structure: Optional[List[str]] = None
    ) -> ChatPromptTemplate:
        """Template for specialized queries requiring specific handling"""
        default_warning = "You are being asked about sensitive information."
        default_requirements = [
            "Provide balanced information",
            "Note important considerations",
            "Include appropriate disclaimers"
        ]
        default_example = [
            "- Key information: [...]",
            "- Important considerations: [...]",
            "- Disclaimer: [...]"
        ]
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            {warning or default_warning}
            You MUST:
            1. {'\n2. '.join(requirements or default_requirements)}
            
            Example structure:
            {'\n'.join(example_structure or default_example)}
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ])
    
    @staticmethod
    def get_citation_format(sources: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Standard format for citations across domains"""
        default_sources = {
            "government": "[{agency} {document_type} {year}]",
            "academic": "[{first_author} et al., {journal} {year}]",
            "organization": "[{org_name} {report_type} {year}]",
            "default": "[{source} {year}]"
        }
        return sources or default_sources
    
    @staticmethod
    def get_disclaimers(custom_disclaimers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Collection of generic disclaimers"""
        default_disclaimers = {
            "sensitive": (
                "**Important**: This information should not replace professional advice. "
                "Consult an appropriate expert for personal guidance."
            ),
            "default": (
                "**Disclaimer**: This information is for educational purposes only. "
                "Always verify with qualified sources."
            )
        }
        return {**default_disclaimers, **(custom_disclaimers or {})}
    
    @staticmethod
    def select_prompt(
        query: str,
        prompt_mappings: Dict[str, ChatPromptTemplate],
        default_prompt: ChatPromptTemplate,
        chat_history: List = None
    ) -> ChatPromptTemplate:
        """Route to appropriate prompt based on query content and provided mappings"""
        query_lower = query.lower()
        
        for keyword, prompt in prompt_mappings.items():
            if keyword in query_lower:
                return prompt
                
        return default_prompt