import os
import sys
from typing import Literal

# --- 1. LANGCHAIN IMPORTS for RAG Components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- GEMINI & GOOGLE AI STUDIO IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Core LangChain component
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import DistanceStrategy

# --- CONFIGURATION ---
PDF_PATH = "sample_1.pdf"

# Indexing Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Parameters (how many chunks to fetch for the LLM)
K_RETRIEVAL = 3 

# --- API CHECK & INITIALIZATION ---
# --- API CHECK & INITIALIZATION ---
def initialize_models():
    """Initializes the Gemini LLM and Embedding Model."""
    
    # Change the variable check to GOOGLE_API_KEY
    if not os.getenv("GOOGLE_API_KEY"):
        print("🚨 ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key to proceed.")
        # Fallback check for GEMINI_API_KEY in case you still want to use it
        if not os.getenv("GEMINI_API_KEY"):
            sys.exit(1)
        else:
            # The LangChain library should pick this up, but the official doc favors GOOGLE_API_KEY
            print("❗ Using GEMINI_API_KEY as a fallback. For reliability, use GOOGLE_API_KEY.")


    # Initialize Embeddings (LangChain's components will look for the key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004" # Powerful, recommended embedding model
    )

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Fast and capable model for RAG
        temperature=0.1
    )
    
    print("✨ Successfully initialized Gemini models for RAG.")
    return llm, embeddings

# --- RAG PROMPT (The Controller) ---
RAG_PROMPT_TEMPLATE = """
You are an expert Q&A system. Your task is to use ONLY the following retrieved context 
to provide a comprehensive and accurate answer to the user's question.

If the answer cannot be found in the context provided, you MUST clearly state: 
"I cannot find the answer in the provided document context." Do not guess or use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""
RAG_PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# --- MAIN AGENT FUNCTION ---
def run_pdf_rag_agent(query: str):
    """Executes the PDF RAG agent pipeline."""
    
    try:
        # Get LLM and Embeddings
        llm, embeddings = initialize_models()

        # --- 1. Load PDF ---
        if not os.path.exists(PDF_PATH):
            print(f"\n🚨 ERROR: File not found at {PDF_PATH}")
            print("Please ensure 'document.pdf' is in the same directory.")
            sys.exit(1)
            
        print(f"\n[1/5] Loading PDF from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages.")

        # --- 2. Chunk Text ---
        print(f"[2/5] Splitting into {len(documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        # --- 3. Create Embeddings & Vector Store (FAISS) ---
        print("[3/5] Generating embeddings and building FAISS index...")
        
        vector_store = FAISS.from_documents(
            chunks, 
            embeddings,
            distance_strategy=DistanceStrategy.COSINE # Recommended for Gemini Embeddings
        )
        print("✅ FAISS Indexing Complete.")

        # --- 4. Retrieval & 5. Generation (RAG Chain) ---
        print(f"[4/5] Setting up Retrieval Chain (k={K_RETRIEVAL})...")
        
        # Create the Retriever component (fetches top K chunks)
        retriever = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVAL})

        # Create the RetrievalQA Chain (The Orchestrator that connects everything)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT}, # Inject the custom RAG prompt
            return_source_documents=True 
        )

        # --- Run the Query ---
        print(f"[5/5] Invoking Agent with Query: '{query}'")
        print("=" * 70)
        
        result = qa_chain.invoke({"query": query})

        # --- Output Results ---
        print("\n--- GEMINI RAG AGENT RESPONSE ---")
        print(result['result'])
        print("\n--- CONTEXT SOURCES (RETRIEVED PASSAGES) ---")
        
        source_docs = result['source_documents']
        for i, doc in enumerate(source_docs):
            page_content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
            print(f"\n[Source {i+1}] File: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Snippet: {page_content_preview}")
        
        print("=" * 70)

    except Exception as e:
        print(f"\n🚨 A critical error occurred: {e}")
        print("Please double-check your API key and file path.")


if __name__ == "__main__":
    # Define the question your agent should answer
    user_query = "summarise it"
    
    run_pdf_rag_agent(user_query)