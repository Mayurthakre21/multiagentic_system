import os
import json
import logging
import uuid
from flask import Flask, request, jsonify, send_from_directory, make_response
from dotenv import load_dotenv
from datetime import datetime
# Import Agents
# NOTE: Ensure you have placed the provided code in the 'agents' directory
from agents_m.pdf_rag import PdfRagAgent
from agents_m.web_search import WebSearchAgent
from agents_m.arxiv_agent import ArXivAgent
from agents_m.controller import ControllerAgent

# --- CONFIGURATION ---
load_dotenv()
UPLOAD_FOLDER = 'uploaded_pdfs'
LOGS_FILE = 'logs/decision_log.json'
ALLOWED_EXTENSIONS = {'pdf'}

# --- FLASK APP SETUP ---
app = Flask(__name__, static_folder='frontend', static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOGGING SETUP (Global) ---
# Set up a general log for console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

# Set up a specific log for decision tracking (will write to JSON file)
def log_decision(trace: dict):
    """Appends a trace dictionary to the decision log JSON file."""
    try:
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        if not os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'w') as f:
                json.dump([], f) # Initialize as an empty list
                
        with open(LOGS_FILE, 'r+') as f:
            data = json.load(f)
            data.insert(0, trace) # Prepend the newest log
            f.seek(0) # Rewind
            json.dump(data, f, indent=4)
    except Exception as e:
        app_logger.error(f"Failed to write to decision log: {e}")

# --- AGENT INITIALIZATION ---
try:
    # 1. Instantiate Core Agents
    pdf_rag_agent = PdfRagAgent()
    web_search_agent = WebSearchAgent()
    arxiv_agent = ArXivAgent()

    # 2. Map Agents for the Controller
    specialized_agents = {
        "pdf_rag": pdf_rag_agent,
        "web_search": web_search_agent,
        "arxiv": arxiv_agent,
        # Add other agents here if you expand the system
    }

    # 3. Instantiate Controller
    controller = ControllerAgent(specialized_agents)
    app_logger.info("All agents and Controller initialized successfully.")

except EnvironmentError as e:
    app_logger.error(f"CRITICAL INIT ERROR: {e}. Exiting.")
    exit(1)
except Exception as e:
    app_logger.error(f"CRITICAL INIT ERROR: An unexpected error occurred during agent setup: {e}")
    exit(1)


# --- UTILITY FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FLASK ENDPOINTS ---

@app.route('/')
def serve_index():
    """Serves the main frontend page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/ask', methods=['POST'])
def ask_query():
    """
    Main endpoint for user queries. The Controller decides which agent to call.
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"answer": "Query cannot be empty.", "agents_used": "None"}), 400

    app_logger.info(f"Received query: '{query}'")

    # The actual call will be wrapped in a function that logs the decision
    def execute_and_log():
        log_entry = {
            "id": str(uuid.uuid4()),
            # FIXED LINE: Use datetime.now() for the timestamp
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], 
            "input_query": query,
            "controller_reasoning": "Determined by LLM-based routing logic (Agent selection is the reasoning).", 
            "agents_called": "",
            "docs_retrieved_info": "",
            "final_synthesized_answer": ""
        }
        
        try:
            # 1. Determine target agent
            routing_chain = controller.ROUTER_PROMPT | controller.router_llm
            target_agent_name = routing_chain.invoke({
                "agent_names": ", ".join(controller.agent_names), 
                "query": query
            }).content.strip().lower().replace('"', '') 
            
            # Fallback for LLM hallucination
            if target_agent_name not in controller.agents:
                target_agent_name = 'web_search'
                
            log_entry['agents_called'] = target_agent_name
            app_logger.info(f"Controller routed to: {target_agent_name}")
            
            # 2. Run the selected agent
            selected_agent = controller.agents[target_agent_name]
            raw_result = selected_agent.run(query)
            
            final_answer = raw_result

            # 3. Enhance Log for PDF RAG Agent
            if target_agent_name == 'pdf_rag' and pdf_rag_agent.vector_store:
            # This line will now succeed because pdf_rag_agent now has K_RETRIEVAL
                log_entry['docs_retrieved_info'] = f"RAG search performed on document: {pdf_rag_agent.current_pdf_name}. K={pdf_rag_agent.K_RETRIEVAL} chunks retrieved."
            elif target_agent_name == 'pdf_rag':
                log_entry['docs_retrieved_info'] = "RAG search attempted, but no PDF was loaded."
            
            # 4. Finalize Log and Response
            log_entry['final_synthesized_answer'] = final_answer[:500] + "..." if len(final_answer) > 500 else final_answer
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
                "answer": f"A system error occurred. Please check the backend logs. Details: {e}",
                "agents_used": "System Failure",
                "decision_rationale": "Execution failed."
            }), 500
    
    return execute_and_log()


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Endpoint to handle PDF uploads and process them for the RAG agent."""
    
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
        
    if file and allowed_file(file.filename):
        # Create unique filename to prevent conflicts
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Ensure upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            file.save(filepath)
            
            # Process the PDF using the PdfRagAgent
            app_logger.info(f"Starting RAG processing for: {filename}")
            result = pdf_rag_agent.process_pdf(filepath)
            
            # Clean up the uploaded file after processing
            # os.remove(filepath)
            
            return jsonify(result), 200

        except Exception as e:
            app_logger.error(f"Error during PDF upload or processing: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Server error during processing: {e}"}), 500
            
    return jsonify({"status": "error", "message": "Invalid file type. Only PDF is allowed."}), 400


@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve the decision logs."""
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'r') as f:
                logs = json.load(f)
            
            # Return as JSON
            return jsonify(logs), 200
        else:
            return jsonify({"message": "No decision logs available yet."}), 200
    except Exception as e:
        app_logger.error(f"Error reading logs file: {e}")
        return jsonify({"message": f"Error reading logs: {e}"}), 500


if __name__ == '__main__':
    # Flask runs on port 5000 by default
    app_logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)