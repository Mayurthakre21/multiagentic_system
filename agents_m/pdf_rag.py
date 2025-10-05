import os
import logging
import sys
from typing import Literal

# --- LANGCHAIN IMPORTS for RAG Components ---
# NOTE: Ensure you have langchain, langchain-community, and google-genai installed
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- GEMINI & GOOGLE AI STUDIO IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Core LangChain component
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import DistanceStrategy

# --- LOGGING SETUP ---
logger = logging.getLogger("PdfRagAgent")

# --- CONFIGURATION (Agent-specific) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 3 

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


class PdfRagAgent:
    """
    Agent responsible for handling PDF uploads, building a vector store, 
    and performing Retrieval-Augmented Generation (RAG) against it.
    """
    def __init__(self):
        """Initializes the LLM, Embeddings, and an empty vector store placeholder."""
        
        # 1. API Key Check (LangChain components will check for GOOGLE_API_KEY or GEMINI_API_KEY)
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            logger.error("🚨 ERROR: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.")
            raise EnvironmentError("Missing API key for Google Generative AI.")

        # 2. Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004" 
        )

        # 3. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1
        )
        
        # 4. State Management
        self.vector_store = None
        self.current_pdf_name = "None"
        
        # --- FIX: Make the global constant an accessible instance attribute ---
        self.K_RETRIEVAL = K_RETRIEVAL 
        
        logger.info("✨ PdfRagAgent initialized successfully with Gemini models.")

    def process_pdf(self, filepath: str):
        """
        Loads, chunks, and indexes the PDF file, updating the agent's internal vector store.
        This is called by the Flask '/api/upload_pdf' endpoint.
        """
        if not os.path.exists(filepath):
            logger.error(f"🚨 File not found at {filepath}")
            return {"status": "error", "message": "File not found."}

        self.current_pdf_name = os.path.basename(filepath)
        
        try:
            # --- 1. Load PDF ---
            logger.info(f"[1/5] Loading PDF from: {self.current_pdf_name}")
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages.")

            # --- 2. Chunk Text ---
            logger.info(f"[2/5] Splitting {len(documents)} pages into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks.")

            # --- 3. Create Embeddings & Vector Store (FAISS) ---
            logger.info("[3/5] Generating embeddings and building FAISS index...")
            
            # This creates the vector store and stores it in the agent instance
            self.vector_store = FAISS.from_documents(
                chunks, 
                self.embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            logger.info("✅ FAISS Indexing Complete.")
            
            return {
                "status": "success", 
                "message": f"Knowledge base loaded from PDF: {self.current_pdf_name}. Ready to answer questions."
            }

        except Exception as e:
            self.vector_store = None # Clear state on failure
            logger.error(f"🚨 A critical error occurred during PDF processing: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to process PDF: {e}"}


    def run(self, query: str):
        """
        Executes the RAG chain on the current vector store with the given query.
        This is called by the Flask '/api/chat' endpoint via the Controller.
        """
        if not self.vector_store:
            return (
                f"Agent 1 (PDF RAG) used the knowledge base ({self.current_pdf_name}). "
                f"Response: No document has been successfully loaded yet. "
                f"Please upload a PDF first."
            )

        try:
            # --- 4. Retrieval & 5. Generation (RAG Chain) ---
            logger.info(f"[4/5] Setting up Retrieval Chain (k={self.K_RETRIEVAL})...") 
            
            # Create the Retriever component (fetches top K chunks)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": self.K_RETRIEVAL}) 
            
            # Create the RetrievalQA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", 
                retriever=retriever,
                chain_type_kwargs={"prompt": RAG_PROMPT},
                return_source_documents=True 
            )

            # --- Run the Query ---
            logger.info(f"[5/5] Invoking RAG Agent with Query: '{query}' on {self.current_pdf_name}")
            result = qa_chain.invoke({"query": query})

            final_response = result['result']
            
            # Include source details in the response for transparency (optional)
            sources = []
            for doc in result.get('source_documents', []):
                sources.append(
                    f"[Page {doc.metadata.get('page', 'N/A') + 1}] Snippet: " 
                    f"{doc.page_content[:150].replace('\n', ' ')}..."
                )

            source_text = "\n\n**Sources from Document Context**:\n" + "\n".join(sources) if sources else ""

            return f"{final_response}{source_text}"

        except Exception as e:
            logger.error(f"🚨 Error during RAG query execution: {e}", exc_info=True)
            return "An internal error occurred while querying the document."


def run_pdf_rag_agent():
    """Factory function required by app.py to instantiate the agent."""
    return PdfRagAgent()
