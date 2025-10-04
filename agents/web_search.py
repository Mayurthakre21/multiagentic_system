# 1. Imports and Environment Setup
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun 
from langchain_google_genai import ChatGoogleGenerativeAI 

# Load environment variables from .env file
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# 2. Define the Search Tool 
search_tool = DuckDuckGoSearchRun(
    name="Web Search", 
    description="A useful tool for searching the internet to get up-to-date information. Returns snippets, links, and titles."
)
tools = [search_tool]

# 3. Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
)

# 4. Define the Custom Agent Prompt (FIXED: Changed Final Answer to standard format)
# Note the change from 'Action: Final Answer' to 'Final Answer:'
CUSTOM_PROMPT = """
You are a top-tier Web Research Agent powered by Google Gemini. Your goal is to 
answer the user's question accurately by using the provided 'Web Search' tool.

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
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    # Keep handle_parsing_errors=True for general robustness
    handle_parsing_errors=True 
)

# 6. Run the Agent
user_query = "what is demon slayer world wide collection ?"

print(f"\n--- Running Gemini Agent for Query: '{user_query}' ---")
try:
    # Running the agent
    result = agent_executor.invoke({"input": user_query})
    
    print("\n--- FINAL ANSWER ---")
    print(result['output'])

except Exception as e:
    # A cleaner error message if something goes wrong outside the agent's control
    print(f"\nAn unhandled error occurred during agent execution: {e}")