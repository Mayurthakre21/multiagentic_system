import arxiv
import pdfplumber
import os
import logging
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# --- LOGGING SETUP ---
logger = logging.getLogger("ArXivAgent")

# --- CONFIGURATION (Agent-specific) ---
MODEL_NAME = "facebook/bart-large-cnn"
TEMP_PDF_DIR = "./temp_pdfs"
MAX_PAPERS = 3 
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30


class ArXivAgent:
    """
    A specialized agent that searches ArXiv, downloads PDFs, extracts text,
    and generates deep summaries using a BART transformer model.
    """
    
    def __init__(self):
        """Initializes the BART summarization model, tokenizer, and device."""
        
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        # Initialize the summarization model and tokenizer
        try:
            logger.info("Initializing BART summarization model...")
            self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
            
            # Use GPU if available (recommended for speed)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            if not os.path.exists(TEMP_PDF_DIR):
                os.makedirs(TEMP_PDF_DIR)
                
            logger.info(f"✨ BART Model loaded successfully on {self.device}.")

        except Exception as e:
            logger.error(f"🚨 Error loading model: {e}")
            self.model = None 
            self.tokenizer = None
            logger.warning("ArXiv deep summarization disabled due to model loading error.")


    def _search_arxiv(self, query, max_results=MAX_PAPERS):
        """Searches ArXiv and returns a list of result metadata."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate 
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title,
                'id': result.entry_id.split('/abs/')[-1],
                'authors': [author.name for author in result.authors],
                'summary_text': result.summary, 
                'result_object': result 
            })
        return papers

    def _extract_text_from_pdf(self, pdf_path):
        """Extracts text from a local PDF file using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
            
            # Simple cleanup
            text = text.replace('\n', ' ').replace('  ', ' ')
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def _summarize_text(self, text):
        """Generates a summary using the BART model."""
        if not self.tokenizer or not self.model:
            return "Summarization model not initialized or failed to load."
        
        try:
            # IMPORTANT: BART has a max input of 1024 tokens. Truncate text to fit.
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(self.device)
            
            # Generate Summary
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=MAX_SUMMARY_LENGTH, 
                min_length=MIN_SUMMARY_LENGTH, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error during BART summarization: {e}")
            return f"Error: Failed to generate deep summary due to model execution error."

    def _format_results(self, query: str, results: list) -> str:
        """Formats the list of paper results into a single, clean Markdown string."""
        if not results:
            return f"The ArXiv Agent found no relevant papers for the query: '{query}'."

        output = [
            f"**ArXiv Search Agent Report**\n",
            f"Successfully found and processed {len(results)} papers for query: '{query}'."
        ]
        
        for i, paper in enumerate(results):
            # Clean up the deep summary slightly for better display
            deep_summary = paper['Agent Deep Summary (Full Paper)'].replace('\n', ' ')
            
            output.append(
                f"\n---\n"
                f"### {i+1}. {paper['Title']}\n"
                f"**Authors:** {paper['Authors']}\n"
                f"**ArXiv Link:** {paper['ArXiv Link']}\n"
                f"**ArXiv Abstract (Source Summary):** {paper['ArXiv Abstract (Source Summary)']}\n"
                f"**Agent Deep Summary (Full Paper):** {deep_summary}"
            )
            
        return "\n".join(output)


    def run(self, query: str, num_papers: int = MAX_PAPERS) -> str:
        """
        Main execution method called by the Controller. 
        Searches, downloads, extracts, and summarizes papers, then formats the output.
        """
        logger.info(f"🚀 ArXiv Agent starting search for: '{query}' (Max {num_papers} papers).")
        papers = self._search_arxiv(query, max_results=num_papers)

        if not papers:
            return self._format_results(query, [])

        results = []
        for i, paper in enumerate(papers):
            logger.info(f"--- [{i+1}/{len(papers)}] Processing: {paper['title']}")
            
            pdf_path = os.path.join(TEMP_PDF_DIR, f"{paper['id']}.pdf")
            deep_summary = "N/A"

            try:
                # 1. Download the PDF
                logger.debug("   - Downloading PDF...")
                paper['result_object'].download_pdf(dirpath=TEMP_PDF_DIR, filename=f"{paper['id']}.pdf")
                
                # 2. Extract Text
                logger.debug("   - Extracting full text...")
                full_text = self._extract_text_from_pdf(pdf_path)
                
                # 3. Deep Summary
                if self.model and full_text and len(full_text) > 100:
                    logger.debug("   - Generating deep summary...")
                    deep_summary = self._summarize_text(full_text)
                elif not self.model:
                    deep_summary = "Deep summarization skipped (BART model not available)."
                elif full_text:
                    deep_summary = "Full text extracted, but too short for reliable deep summarization."
                else:
                    deep_summary = "Could not reliably extract full text for deep summary."
                    
            except Exception as e:
                logger.error(f"   - ERROR during download/summarization for {paper['id']}: {e}")
                deep_summary = f"Error during processing: {e}"
            
            finally:
                # Clean up the downloaded PDF file
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    logger.debug(f"Cleaned up temporary file: {pdf_path}")

            # Compile the final result for formatting
            results.append({
                'Title': paper['title'],
                'Authors': ", ".join(paper['authors']),
                'ArXiv Link': f"https://arxiv.org/abs/{paper['id']}",
                'ArXiv Abstract (Source Summary)': paper['summary_text'],
                'Agent Deep Summary (Full Paper)': deep_summary
            })
            
        return self._format_results(query, results)

# NOTE: No factory function is needed here, as app.py instantiates this class directly:
# "arxiv": ArXivAgent(),