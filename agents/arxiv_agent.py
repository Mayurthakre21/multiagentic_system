import arxiv
import pdfplumber
import os
import json
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# --- 1. SETUP AND INITIALIZATION ---

# Initialize the summarization model and tokenizer
# BART-large-CNN is excellent for abstractive summarization
try:
    print("Initializing BART summarization model (first time may download files)...")
    MODEL_NAME = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Use GPU if available (recommended for speed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    print(f"Model loaded successfully on {DEVICE}.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Check your 'torch' and 'transformers' installation.")
    tokenizer, model = None, None # Set to None if initialization fails


# --- 2. CORE FUNCTIONS ---

def search_arxiv(query, max_results=5):
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
            'id': result.entry_id.split('/abs/')[-1], # e.g., '2405.00001'
            'authors': [author.name for author in result.authors],
            'summary_text': result.summary, # ArXiv provides the abstract
            'result_object': result # Keep the full object for easy download
        })
    return papers

def extract_text_from_pdf(pdf_path):
    """Extracts text from a local PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Basic text extraction from the page
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        
        # Simple cleanup: replace multiple newlines/spaces
        text = text.replace('\n', ' ').replace('  ', ' ')
        return text
    except Exception as e:
        # print(f"Error extracting text from {pdf_path}: {e}")
        return None

def summarize_text(text, max_length=150, min_length=30):
    """Generates a summary using the BART model."""
    if not tokenizer or not model:
        return "Summarization model not initialized."
    
    # IMPORTANT: BART has a max input of 1024 tokens. Truncate text to fit.
    # We are summarizing the start of the paper (Intro/Methods).
    inputs = tokenizer(text, return_tensors="pt", 
                       max_length=1024, truncation=True).to(DEVICE)
    
    # Generate Summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# --- 3. AGENT ORCHESTRATION ---

def arxiv_agent(query, num_papers=3):
    """
    Main ArXiv Agent: Searches, downloads, extracts, and summarizes papers.
    """
    if not os.path.exists("./temp_pdfs"):
        os.makedirs("./temp_pdfs")
        
    print(f"\n🚀 Searching ArXiv for: '{query}'...")
    papers = search_arxiv(query, max_results=num_papers)

    if not papers:
        return "No relevant papers found.", []

    results = []
    for i, paper in enumerate(papers):
        print(f"\n--- [{i+1}/{len(papers)}] Processing: {paper['title']}")
        
        pdf_path = f"./temp_pdfs/{paper['id']}.pdf"
        deep_summary = "N/A"

        try:
            # 1. Download the PDF
            print("  - Downloading PDF...")
            # The 'result_object' contains the download method
            paper['result_object'].download_pdf(dirpath="./temp_pdfs", filename=f"{paper['id']}.pdf")
            
            # 2. Extract Text
            print("  - Extracting full text...")
            full_text = extract_text_from_pdf(pdf_path)
            
            # 3. Deep Summary
            if full_text and len(full_text) > 100: # Check if extraction was successful
                print("  - Generating deep summary...")
                deep_summary = summarize_text(full_text)
            elif full_text:
                deep_summary = "Full text extracted, but too short for reliable deep summarization."
            else:
                 deep_summary = "Could not reliably extract full text for deep summary."
                
        except Exception as e:
            print(f"  - ERROR during download/summarization: {e}")
            deep_summary = f"Error: {e}"
        
        finally:
            # Clean up the downloaded PDF file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        # Compile the final result
        results.append({
            'Title': paper['title'],
            'Authors': ", ".join(paper['authors']),
            'ArXiv Link': f"https://arxiv.org/abs/{paper['id']}",
            'ArXiv Abstract (Source Summary)': paper['summary_text'],
            'Agent Deep Summary (Full Paper)': deep_summary
        })
        
    return "Agent execution complete.", results


# --- 4. EXECUTION ---

if __name__ == '__main__':
    # Define your search query here
    SEARCH_QUERY = "crop yield prediction using satellite images"
    NUMBER_OF_PAPERS = 2 

    status_message, agent_results = arxiv_agent(SEARCH_QUERY, NUMBER_OF_PAPERS)
    
    print("\n" + "="*80)
    print(f"FINAL AGENT REPORT: {status_message}")
    print("="*80)

    # Print results in a readable JSON format
    print(json.dumps(agent_results, indent=4))
    print("\n" + "="*80)