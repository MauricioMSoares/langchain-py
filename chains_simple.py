from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SimpleSequentialChain, SequentialChain


# Model set-up
chat = ChatOpenAI(temperature=0.0)


# Simple LLM Chain
prompt = ChatPromptTemplate.from_template("Translate this text to Spanish: {text}")
chain = LLMChain(llm=chat, prompt=prompt)
chain.run("Can you translate this sentence?")


# Simple Sequential Chain
prompt_2 = ChatPromptTemplate.from_template("Translate this text to Japanese: {text_2}")
prompt_3 = ChatPromptTemplate.from_template("Translate this text to English: {text_3}")
chain_1 = LLMChain(llm=chat, prompt=prompt)
chain_2 = LLMChain(llm=chat, prompt=prompt_2)
chain_3 = LLMChain(llm=chat, prompt=prompt_3)

simple_sequential_chain = SimpleSequentialChain(
    chains=[chain_1, chain_2, chain_3], verbose=True
)
simple_sequential_chain.run("Can you translate this sentence?")
