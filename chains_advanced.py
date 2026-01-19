from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SimpleSequentialChain, SequentialChain


# Model set-up
chat = ChatOpenAI(temperature=0.0)
prompt = ChatPromptTemplate.from_template("Translate this text to Spanish: {text}")


# Advanced Sequential Chain
chain_1 = LLMChain(llm=chat, prompt=prompt, output_key="text_1")
chain_2 = LLMChain(llm=chat, prompt=prompt, output_key="text_2")
chain_3 = LLMChain(llm=chat, prompt=prompt, output_key="text_3")

prompt_4 = ChatPromptTemplate.from_template(
    """Answer 'Yes' if the following two sentences are the exact same, else answer 'No':

    Sentence 1: {text}
    Sentence 2: {text_3}
    """
)

chain_4 = LLMChain(llm=chat, prompt=prompt_4, output_key="evaluation")

advanced_sequential_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["text"],
    output_variables=["text", "text_3", "evaluation"],
    verbose=True
)

advanced_sequential_chain.run("Can you translate this sentence?")
