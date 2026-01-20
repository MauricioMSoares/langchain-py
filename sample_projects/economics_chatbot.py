import os

os.environ["OPENAI_API_KEY"] = ""
import requests
import json
from datetime import date
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent


prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()


@tool
def pull_economic_data(economic_indicator, start_year, end_year):
    """
    Pull historical economic data.

    Args:
        economic_indicator: Which economic indicator you would like to pull historical data on. This can be one of: CPI, PPI, or unemployment_rate
        start_year: the start year of the date range over which you want to pull economic data
        end_year: the end year of the date range over which you want to pull economic data
    """

    # See this for series ids: https://data.bls.gov/cgi-bin/surveymost?bls
    series_map = {
        "CPI": "CUUR0000SA0",
        "PPI": "WPUFD4",
        "unemployment_rate": "LNS14000000",
    }

    resp = requests.post(
        url="https://api.bls.gov/publicAPI/v1/timeseries/data/",
        data=json.dumps(
            {
                "seriesid": [series_map.get(economic_indicator)],
                "startyear": start_year,
                "endyear": end_year,
            }
        ),
        headers={"Content-type": "application/json"},
    )

    data = json.loads(resp.text)

    try:
        # refine it
        refined_data = []
        for series in data["Results"]["series"]:
            for item in series["data"]:
                refined_data.append(item)

    except:
        refined_data = []

    return refined_data


# params = {'economic_indicator': 'unemployment_rate', 'start_year': 2023, 'end_year': 2024}
# data = pull_economic_data(economic_indicator=params['economic_indicator'], start_year='2015', end_year='2024')

# for item in data[:5]:
#   print(item)

prompt[0].prompt.template = (
    f"Your job is to answer users' economic questions by referencing economic data. Note that today's date is {str(date.today())}"
)

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
agent = create_openai_tools_agent(model, [pull_economic_data], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[pull_economic_data], verbose=True)

response = agent_executor.invoke(
    {
        "input": "Pull unemployment rate data for 2023 and 2024"
    }
)
