from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    evidence: List[str]
    conversation_id: Optional[str] = None
    chat_history: Optional[List[dict]] = None