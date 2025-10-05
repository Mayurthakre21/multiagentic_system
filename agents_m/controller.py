import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

logger = logging.getLogger("ControllerAgent")

class ControllerAgent:
    """
    The main routing agent that takes a user query, uses Gemini to classify 
    the intent, and routes the query to the correct specialized agent.
    """
    def __init__(self, specialized_agents: Dict[str, Any]):
        """
        Initializes the controller with a dictionary of specialized agents.
        """
        self.agents = specialized_agents
        
        # Initialize the router LLM (Gemini 2.5 Flash)
        self.router_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1
        )
        
        # Available agent names for the router prompt
        self.agent_names = list(self.agents.keys())
        
        # Define the prompt for the router LLM
        self.ROUTER_PROMPT = PromptTemplate.from_template("""
            You are a highly intelligent routing agent. Your task is to analyze the user's 
            query and determine the single best specialized agent to handle it.
            
            Available Agents: {agent_names}
            
            - 'web_search': For current events, general knowledge, or anything requiring up-to-date web information.
            - 'arxiv': For deep academic or scientific research, seeking papers, studies, or technical reports.
            - 'pdf_rag': For questions about uploaded documents (assume a relevant document is available).
            
            Your response MUST be ONLY the name of the chosen agent from the list: {agent_names}.
            Do not include any other text, punctuation, or explanation.
            
            User Query: {query}
        """)
        
        logger.info(f"Controller initialized with agents: {self.agent_names}")

    def route_query(self, query: str) -> str:
        """
        Routes the user query to the appropriate agent and returns the result.
        """
        
        # 1. Determine the target agent using the router LLM
        routing_chain = self.ROUTER_PROMPT | self.router_llm
        
        try:
            # LLM output should be a single agent name string
            target_agent_name = routing_chain.invoke({
                "agent_names": ", ".join(self.agent_names), 
                "query": query
            }).content.strip().lower().replace('"', '') # Clean up the output
            
            if target_agent_name not in self.agents:
                logger.warning(f"Router returned invalid agent name: {target_agent_name}. Falling back to 'web_search'.")
                target_agent_name = 'web_search'
                
            logger.info(f"Query routed to agent: '{target_agent_name}'")

            # 2. Run the selected agent
            selected_agent = self.agents[target_agent_name]
            
            # All specialized agents must have a .run(query) method
            result = selected_agent.run(query)
            
            return result

        except Exception as e:
            logger.error(f"Error during query routing or agent execution: {e}")
            return f"Error processing request: Failed to route or execute agent. ({e})"

# NOTE: The initialization of this class is handled in app.py
