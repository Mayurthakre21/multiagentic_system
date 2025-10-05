import os
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun 
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

# Load environment variables (needed here in case this agent is tested standalone)
load_dotenv()

logger = logging.getLogger("WebSearchAgent")

class WebSearchAgent:
    """
    A specialized agent that performs web research using LangChain's ReAct framework
    and the DuckDuckGo search tool, powered by Google Gemini.
    """
    
    def __init__(self):
        """Initializes the LLM, tool, prompt, and the Agent Executor (RAG Chain)."""
        
        # 1. API Key Check
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("🚨 GOOGLE_API_KEY not found. WebSearchAgent failed to initialize.")
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        # 2. Define the Search Tool 
        self.search_tool = DuckDuckGoSearchRun(
            name="Web_Search_Tool", 
            description="A useful tool for searching the internet to get up-to-date information. Returns snippets, links, and titles."
        )
        self.tools = [self.search_tool]

        # 3. Initialize the Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
        )

        # 4. Define the Custom Agent Prompt
        CUSTOM_PROMPT = """
        You are a top-tier Web Research Agent powered by Google Gemini. Your goal is to 
        answer the user's question accurately by using the provided 'Web_Search_Tool' tool.

        After you have gathered information from the tool, you MUST synthesize the top 3-5 results 
        into a concise, comprehensive, and well-structured summary. 
        Always include the source titles and URLs for the top 3 results in your final answer.

        You MUST use the following format for your response:

        Thought: You should always think about what to do next.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action

        ... (when you have the final answer)

        Thought: I have gathered enough information and will now provide the final answer.
        Final Answer: The comprehensive summary and source links.

        TOOLS:
        ------
        {tools}
        Available Tools: {tool_names} 

        {agent_scratchpad}

        User's Query: {input}
        """
        prompt = PromptTemplate.from_template(CUSTOM_PROMPT)

        # 5. Create the Agent Executor
        agent = create_react_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False, # Set to True for detailed console logs
            handle_parsing_errors=True 
        )
        
        logger.info("WebSearchAgent initialized successfully.")

    def run(self, query: str) -> str:
        """
        Executes the web search agent with the given query.
        This is called by the Flask '/api/chat' endpoint via the Controller.
        """
        logger.info(f"Invoking Web Search Agent for query: '{query}'")
        
        try:
            # 6. Run the Agent
            result = self.agent_executor.invoke({"input": query})
            
            # The 'output' key contains the result formulated by the LLM
            final_answer = result['output']
            
            # Log the successful execution (only final answer)
            logger.info("Web Search Agent executed successfully.")
            
            # Return the synthesized result string
            return final_answer

        except Exception as e:
            logger.error(f"An error occurred during WebSearchAgent execution: {e}", exc_info=True)
            # Return a controlled error message
            return f"Web Search Agent failed to execute the query. Error details: {e}"

# NOTE: No factory function is needed here, as app.py instantiates this class directly:
# "web_search": WebSearchAgent(),