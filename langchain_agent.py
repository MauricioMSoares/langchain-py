import os
os.environ["OPENAI_API_KEY"] = ""
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.agents import load_tools, initialize_agent, AgentType


llm = ChatOpenAI(temperature=0.0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("How old is Michael Jordan? How long has it been since he retired from basketball?")
