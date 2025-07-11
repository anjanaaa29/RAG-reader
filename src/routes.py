from fastapi import APIRouter, HTTPException
from retrieval_chain import RetrievalChain
from utils import ChatUtils  # Assuming you have this for history management
from schemas import ChatRequest, ChatResponse

router = APIRouter()
retrieval_chain = RetrievalChain()

# Simple in-memory store (replace with Redis/DB in production)
conversation_store = {}

def get_history(conversation_id: str):
    """Retrieve conversation history from store"""
    if not conversation_id:
        return []
    return conversation_store.get(conversation_id, [])

def save_history(conversation_id: str, query: str, answer: str):
    """Update conversation history in store"""
    if not conversation_id:
        return
        
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []
    
    conversation_store[conversation_id].extend([
        {"user": query},
        {"bot": answer}
    ])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Get existing history
        history = get_history(request.conversation_id)
        
        # Process query
        result = retrieval_chain.invoke({
            "input": request.query,
            "chat_history": history
        })
        
        answer = result.get("answer", "No answer generated")
        sources = [doc.metadata.get("source", "unknown") for doc in result["context_documents"][:7]]
        evidence = [doc.page_content[:250] + "..." for doc in result["context_documents"][:7]]
        
        # Generate new conversation_id if first message
        conversation_id = request.conversation_id or ChatUtils.generate_conversation_id()
        
        # Save conversation
        save_history(conversation_id, request.query, answer)
        
        return {
            "answer": answer,
            "sources": sources,
            "evidence": evidence,
            "conversation_id": conversation_id,
            "chat_history": get_history(conversation_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))