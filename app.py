import streamlit as st
import os
from streamlit_chat import message
import tempfile

# Import your RAG modules directly
from src.doc_loader import DocumentLoader
from src.vector_db import VectorDB
from src.retrieval_chain import RetrievalChain
from src.citation import GenericCitationFormatter
from src.utils import ChatUtils

st.set_page_config(page_title="📚 RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []  # [(user, bot)]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# --- Sidebar: Upload & Clear ---
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Select files to upload and index",
    type=["pdf", "docx", "txt", "csv", "json"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please select at least one file.")
    else:
        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        for file in uploaded_files:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(path)

        with st.spinner("Processing documents..."):
            try:
                loader = DocumentLoader()
                documents = []
                for path in temp_paths:
                    documents.extend(loader.load_document(path))

                vector_db = VectorDB()
                vectorstore = vector_db.create_from_documents(documents)

                retrieval_system = RetrievalChain()
                retriever = retrieval_system.get_retriever(vectorstore)
                retrieval_chain = retrieval_system.create_retrieval_chain(retriever)

                st.session_state.vectorstore = vectorstore
                st.session_state.retrieval_chain = retrieval_chain
                st.session_state.docs_processed = True
                st.sidebar.success(f"✅ {len(uploaded_files)} documents indexed.")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.sidebar.info("Chat history cleared.")

st.markdown("---")

# --- Chat Section ---
st.header("💬 Chat with your documents")

query = st.chat_input("Ask a question about the documents…")

def generate_response(query: str) -> str:
    if not st.session_state.docs_processed:
        return "⚠️ Please upload and process documents first."

    try:
        result = st.session_state.retrieval_chain.invoke({
            "input": query,
            "chat_history": st.session_state.history
        })

        answer = result.get("answer", "No answer.")
        context_docs = result.get("context_documents", [])

        sources = []
        excerpts = []
        for doc in context_docs:
            citation = GenericCitationFormatter.format_citation(doc.metadata)
            sources.append(citation)
            content = doc.page_content
            excerpt = ChatUtils.clean_text(content[:250] + "...") if len(content) > 250 else content
            excerpts.append(excerpt)

        disclaimer = ChatUtils.generate_disclaimer(answer, "general")

        # Simplified response format
        full_response = f"{answer}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:3])
        if excerpts:
            full_response += "\n\nRelevant excerpts:\n" + "\n".join(f"- {e}" for e in excerpts[:3])
        full_response += f"\n\n{disclaimer}"
        
        return full_response

    except Exception as e:
        return f"Error: {str(e)}"

if query:
    with st.spinner("Thinking..."):
        response = generate_response(query)
        st.session_state.history.append(
            (query, response)
        )

# --- Display Chat History ---
if not st.session_state.history:
    st.info("💬 Start by asking a question!")
else:
    for idx, (q, a) in enumerate(st.session_state.history):
        message(q, is_user=True, key=f"user_{idx}")
        message(a, key=f"bot_{idx}")
