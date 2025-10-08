import os
import json
import pandas as pd
import logging
import uuid
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from datetime import datetime
from flask_cors import CORS

# --- IMPORT AGENTS ---
from agents_m.pdf_rag import PdfRagAgent
from agents_m.web_search import WebSearchAgent
from agents_m.arxiv_agent import ArXivAgent
from agents_m.controller import ControllerAgent

# --- HUGGING FACE DATASET IMPORTS ---
from datasets import Dataset
from huggingface_hub import HfApi

# --- CONFIGURATION ---
load_dotenv()

UPLOAD_FOLDER = '/tmp/uploaded_pdfs'
LOGS_FILE = '/tmp/decision_log.json'
ALLOWED_EXTENSIONS = {'pdf'}

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET = "Mayurthakre21/multiagent_logs"  

# --- FLASK APP SETUP ---
app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOGGING CONFIG ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
app_logger = logging.getLogger(__name__)

# --- UTILITY: DECISION LOGGING ---
def log_decision(trace: dict):
    try:
        # Ensure timestamp
        trace["timestamp"] = trace.get("timestamp") or datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Keep only the fields you want
        structured_trace = {
            "id": trace.get("id"),
            "timestamp": trace.get("timestamp"),
            "input_query": trace.get("input_query"),
            "controller_reasoning": trace.get("controller_reasoning"),
            "agents_called": trace.get("agents_called"),
            "final_synthesized_answer": trace.get("final_synthesized_answer")
        }

        # Local backup
        os.makedirs(os.path.dirname(LOGS_FILE), exist_ok=True)
        if not os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'w') as f:
                json.dump([], f, indent=4)
        with open(LOGS_FILE, 'r+') as f:
            data = json.load(f)
            data.insert(0, structured_trace)
            f.seek(0)
            json.dump(data, f, indent=4)  # ✅ Pretty print for readability

        # Push to HF dataset as JSON (optional)
        if HF_TOKEN:
            with open("/tmp/latest_log.json", "w") as f:
                json.dump(structured_trace, f, indent=4)
            api = HfApi()
            api.upload_file(
                path_or_fileobj="/tmp/latest_log.json",
                path_in_repo=f"logs/{trace['id']}.json",
                repo_id=HF_DATASET,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            app_logger.info(f"✅ Log pushed to HF dataset: {HF_DATASET}")
        else:
            app_logger.warning("⚠️ HF_TOKEN not found. Skipping dataset push.")

    except Exception as e:
        app_logger.error(f"🚨 Failed to push log: {e}", exc_info=True)

# --- AGENT INITIALIZATION ---
try:
    pdf_rag_agent = PdfRagAgent()
    web_search_agent = WebSearchAgent()
    arxiv_agent = ArXivAgent()

    specialized_agents = {
        "pdf_rag": pdf_rag_agent,
        "web_search": web_search_agent,
        "arxiv": arxiv_agent,
    }

    controller = ControllerAgent(specialized_agents)
    app_logger.info("✅ All agents and Controller initialized successfully.")

except EnvironmentError as e:
    app_logger.error(f"🚨 CRITICAL INIT ERROR: {e}. Exiting.")
    exit(1)
except Exception as e:
    app_logger.error(f"🚨 CRITICAL INIT ERROR: Unexpected error during setup: {e}")
    exit(1)

# --- FILE VALIDATION ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400

    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            app_logger.info(f"📄 PDF uploaded successfully: {filename}")

            result = pdf_rag_agent.process_pdf(filepath)
            pdf_rag_agent.current_pdf_name = filename
            app_logger.info(f"✅ PDF processed and vectorized: {filename}")

            return jsonify(result), 200

        except Exception as e:
            app_logger.error(f"Error during PDF processing: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

    return jsonify({"status": "error", "message": "Invalid file type. Only PDF allowed."}), 400

@app.route('/ask', methods=['POST'])
def ask_query():
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"answer": "Query cannot be empty.", "agents_used": "None"}), 400

    app_logger.info(f"Received query: '{query}'")

    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "input_query": query,
        "controller_reasoning": "Determined by LLM-based routing logic.",
        "agents_called": "",
        "docs_retrieved_info": "",
        "final_synthesized_answer": ""
    }

    try:
        routing_chain = controller.ROUTER_PROMPT | controller.router_llm
        target_agent_name = routing_chain.invoke({
            "agent_names": ", ".join(controller.agent_names),
            "query": query
        }).content.strip().lower().replace('"', '')

        if target_agent_name not in controller.agents:
            target_agent_name = 'web_search'

        log_entry['agents_called'] = target_agent_name
        app_logger.info(f"Controller routed to: {target_agent_name}")

        selected_agent = controller.agents[target_agent_name]
        raw_result = selected_agent.run(query)
        final_answer = raw_result

        if target_agent_name == 'pdf_rag' and getattr(pdf_rag_agent, "vector_store", None):
            log_entry['docs_retrieved_info'] = (
                f"RAG search performed on document: {pdf_rag_agent.current_pdf_name}. "
                f"K={pdf_rag_agent.K_RETRIEVAL} chunks retrieved."
            )
        elif target_agent_name == 'pdf_rag':
            log_entry['docs_retrieved_info'] = "RAG search attempted, but no PDF was loaded."

        log_entry['final_synthesized_answer'] = (
            final_answer[:500] + "..." if len(final_answer) > 500 else final_answer
        )
        log_decision(log_entry)

        return jsonify({
            "answer": final_answer,
            "agents_used": target_agent_name,
            "decision_rationale": "LLM Router chose the best-fit agent."
        }), 200

    except Exception as e:
        error_message = f"Critical error during query execution: {e}"
        app_logger.error(error_message, exc_info=True)
        log_entry['final_synthesized_answer'] = f"System Error: {error_message}"
        log_decision(log_entry)
        return jsonify({
            "answer": f"A system error occurred. Details: {e}",
            "agents_used": "System Failure",
            "decision_rationale": "Execution failed."
        }), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'r') as f:
                logs = json.load(f)
            return jsonify(logs), 200
        return jsonify({"message": "No decision logs available yet."}), 200
    except Exception as e:
        app_logger.error(f"Error reading logs file: {e}")
        return jsonify({"message": f"Error reading logs: {e}"}), 500

# --- APP ENTRYPOINT ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
