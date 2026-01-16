from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


chat = ChatOpenAI(temperature=0.0)
chat

template = """
Extract the following details from a description of a company:

name: the name of the company
sector: the sector it is in
valuation: its valuation in dollars $

Format the output as JSON with the following keys:

name
sector
valuation

company description:

{company_description}
"""

prompt_template = ChatPromptTemplate.from_template(template)
prompt_template.messages[0].input_variables

company_description = """
EcoFusion Technologies is an innovative start-up breaking new grounds in the clean energy sector.
The company focuses on designing and manufacturing highly efficient fusion energy generators that promote green and sustainable energy.
Through the fusion of cutting-edge technology and commitment to environmental preservation, EcoFusion Technologies is pushing the
boundaries of what's possible in the renewable energy landscape.

Current Valuation: As of our latest funding round, EcoFusion Technologies is values at $1.2 billion, demonstrating significant investor 
confidence in our future growth prospects.
"""

filled_in_prompt = prompt_template.format_messages(company_description=company_description)
filled_in_prompt

response = chat(filled_in_prompt).content
print(response)

output_parser = JsonOutputParser(response)
output_parser
