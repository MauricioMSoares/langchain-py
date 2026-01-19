import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.agents import load_tools, initialize_agent, AgentType


llm = ChatOpenAI(temperature=0)
tools = load_tools(["arxiv"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
agent.run("What is the most recent arxiv paper on large language models in the year 2026? What is the key insight from it?")
