import os
from langchain_openai import ChatOpenAI
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, ConfigDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_experimental.utilities import PythonREPL
import matplotlib
from langchain_community.tools.tavily_search import TavilySearchResults
import requests

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=None,
    max_retries=2,
)

@tool
def llm_tool(
    query: Annotated [str,"The query search for."]
): 
    """ A tool to call an llm model to search for q """
    try:
        result = llm.invoke(query)
    except BaseException as e:
        return f"Failed to execute. Error:{repr(e)}"     
    return result.content 

#File Mangements tool 
file_tools = FileManagementToolkit(
    root_dir=str("./data"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = file_tools

# Repl Python code
python_repl = PythonREPL()
@tool 
def python_repl_tool(
    code: Annotated[str,"Python code to execute user insruction to genrate csv or chats."],
):
    """Use this to execute python code and do math. If you want to see the output of a value, 
    you should print it out with `print(...). This is visible to the user. 
    If you need to save a plot, you should save it to the/data folder. 
    Asumme the most default values for charts and plots. If the user has not indicated a prefence, 
    make an assumtiption and create the plot. Do not use a sandboxed environment. 
    Write the files to the ./data folder, residing in the current folder.

    Clean the data provided before plotting a chart. If arrays are of unequal length, 
    substitute missing data points with 0 or the average of the array. 

    Example: 
    Do not save the plot to (sandbox:/data/plot.png) but to (./data/plot.png) 

    Example: 
    from matplotlib import pyplot as plt 
    plt.savefig('./data/foo.png')
    """
    try:
        matplotlib.use('agg')
        result = python_repl.run(code) 
    except BaseException as e:
        return f"Failed to execute.Error {repr(e)}" 

    result_str = f"Successfully executed:\n``` python\n{code}\n```\n stdout:{result}"
    return result_str    

#Tavily serach result 
tavily_tool = TavilySearchResults(max_result = 5)

#Alpha Vantage search tool - provides realtime and historical financial market data
def natural_gas():
    response = requests.get(
        "https://www.alphavantage.co/query/",
        params={
            "function":"NATURAL_GAS",
            "apikey":os.getenv("ALPHAVANTAGE_API_KEY"),
        },
    )
    response.raise_for_status()
    data = response.json()
    
    if "Error message" in data:
        raise ValueError(f"API Error:{data['error messsage']}")
    
    return data

@tool 
def alphavantage_tool():
    "A tool to get natural gas prices through API"
    try:
        result = natural_gas()
    except BaseException as e:
        return f"Failed to execute. Error:{repr(e)}"     
    return result

def get_gdp_data(country_code:str):
    response = requests.get(
        "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH",
    )
    data = response.json()
    if "Error message" in data:
        raise ValueError(f"API Error:{data['error messsage']}") 
    return data["values"]["NGDP_RPCH"][country_code]

@tool
def gdp_tool(country_code:str):
    "A tool to get the gdp data through API"
    try:
        result = get_gdp_data(country_code)
    except BaseException as e:
        return f"Failed to execute. Error:{repr(e)}"     
    return result    

def get_top_headlines():
    url = "https://real-time-news-data.p.rapidapi.com/top-headlines"

    querystring = {"limit":"10"}

    headers = {
        "x-rapidapi-key": "3d4d615b16msh24876e24e8d38d2p15e363jsna7c54675b146",
        "x-rapidapi-host": "real-time-news-data.p.rapidapi.com"
    }

    response = requests.get(url=url,headers=headers,params=querystring)  
    data = response.json() 
    if "Error message" in data:
        raise ValueError(f"API Error:{data['error messsage']}") 
    
    articles = data.get("data", [])
    titles = [article.get("title") for article in articles[:10]]
    
    return titles 

@tool 
def news_tool():
    "A tool to get Top headlines data using API"
    try:
        result = get_top_headlines() 
    except BaseException as e:
        return f"Failed to execute. Error:{repr(e)}"  
    return result     


# Define agent members
members = ["llm","file_writer","coder","researcher","alpha_vantage","gdp_researcher","news_data"]  # Example members - add your actual agents here

# Define agent options as an Enum
class AgentOptions(str, Enum):
    FINISH = "FINISH"
    LLM = "llm"
    FILE_WRITER = "file_writer"
    CODER = "coder"
    RESEARCHER = "researcher"
    ALPHA_VANTAGE = "alpha_vantage"
    GDP_RESEARCHER = "gdp_researcher" 
    NEWS_DATA = "news_data"


# Our custom state class
class Agentstate(MessagesState):
    # next field indicate where to route next
    next: str

# Define the system prompt for the supervisor
# system_prompt = (
#     """
#     You're a supervisor tasked with managing a conversation between workers: {members}.
#     Given the user request, respond with the worker to act next.Each worker will perform
#     the task and respond with their result and status. When Finished, response with FINISH.
#     Always start with the user question and answer with appropriate tools. For File Mangement
#     use the file_writer worker. For Calculations,plots,Charts, CSV file generations use coder
#     worker. for example user ask for bar chart, ask the researcher to gather the data and coder
#     to create and save the plot. Start with the llm tool and call researcher tool only when the 
#     llm results are inadequant. for example , if the user asks for bar chart, ask the researcher
#     to gather the data and the coder to create and save the plot. For research use the
#     research worker. If you asked to get the historical data of Natural gas dont use researcher tool,
#     Go straight to alpha_vantage_tool and ask coder to use the data provided.
#     """.format(members=", ".join(members))
# )

system_prompt = (
    f""" You are a supervisor tasked with managing a conversation between the following workers: {members}.
        Given the following user request, respond with the worker to act next.
        Each worker will perform a task and respond with their results and status.
        When finished, respond with FINISH.

        Always start with the question and see if you can answer it without any tools.
        Then only call the llm tool and then call research tool, only when the LLM tool results are inadequate.
        If no tools are needed, route to llm agent.

        For calculations, visualizations, plots, charts, CSV file generation, use the coder worker.
        For example, if the user asks for a bar chart, ask the researcher to gather the data and then the coder to create and save the plot.

        For GDP research, use the gdp_tool by passing the country code.
        For research, use the researcher worker.
        If you're asked to get historical data for Natural Gas prices, do not use Research tool,
        go straight to alpha_vantage_tool and ask coder to use the data provided. 
        For News data, use the news_tool, do not use the research tool 

        For file management, use the file_writer worker only after the tasks of the researcher and the coder are done
        and if the user requires any file operations.

        FINISH only when all tasks are done.
        Do not end the conversation until all tasks are complete.
        Do not end the conversation until the file is written.
        All files should be written in ./data residing in the current folder.    
    """.format(members=", ".join(members))
)


# Fix: Define Supervisoragent with Pydantic V1 compatibility
class Supervisoragent(BaseModel):
    """Worker to route next; if none, route to FINISH."""
    next: AgentOptions

    # This is the critical fix - adding config to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Nodes
def supervisor_node(state: Agentstate) -> Agentstate:
    print("------------- supervisor state ------------------\n")
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    
    response = llm.with_structured_output(Supervisoragent).invoke(messages)
    next_ = response.next
    
    if next_ == AgentOptions.FINISH:
        next_ = END
    
    print(f"\n Routing to {next_}")
    print("----------------end of supervisor state-----------")
    return {"next": next_}

llm_agent = create_react_agent(
    llm,tools=[llm_tool],state_modifier= """ Your highly-trained research analyst and can provide the user with 
    the information needed. You are tasked with finding the answers to the user questions without using any
    tools. Answer to user question best of your ability."""
)
def llm_node(state: Agentstate) -> Agentstate:
    result = llm_agent.invoke(state)
    return {
        "messages":[
            HumanMessage(content=result['messages'][-1].content,name="llm_node")
        ]
    }

file_agent = create_react_agent(
    llm, tools=[write_tool] 
)
def file_node(state:Agentstate) -> Agentstate:
    result = file_agent.invoke(state)
    return{
        "messages":[HumanMessage(content=result["messages"][-1].content,name="file_writer")]
    }

code_agent = create_react_agent(llm,tools=[python_repl_tool])
def coder_node(state:Agentstate) -> Agentstate:
    result = code_agent.invoke(state)
    return {
        "messages":[HumanMessage(content=result["messages"][-1].content,name="coder")]
    }

research_agent = create_react_agent(
    llm,tools=[tavily_tool],
    state_modifier= """ our highly-trained researcher. Do not do any math. 
    You are tasked with finding the answers to the user questions.You have
    access to following tool : Tavily_search. use wisely.
    """
)
def research_node(state:Agentstate) -> Agentstate:
    result = research_agent.invoke(state)
    return {
        "messages":[HumanMessage(content=result["messages"][-1].content,name="researcher")]
    }

alpha_vantage_agent = create_react_agent(
    llm, tools=[alphavantage_tool]
)
def alpha_vantage_node(state:Agentstate) -> Agentstate:
    result = alpha_vantage_agent.invoke(state)
    return {
        "messages":[HumanMessage(content=result["messages"][-1].content,name="alpha_vantage")]
    }

gdp_agent = create_react_agent(
    llm, tools=[gdp_tool]
)
def gdp_node(state:Agentstate) -> Agentstate:
    result = gdp_agent.invoke(state)
    return {
        "messages":[HumanMessage(content=result["messages"][-1].content,name="gdp_researcher")]
    } 

news_agent = create_react_agent(
    llm,tools=[news_tool] 
)
def news_node(state:Agentstate) -> Agentstate:
    result = news_agent.invoke(state) 
    return {
        "messages":[HumanMessage(content=result['messages'][-1].content,name="news_data")]
    }


# Build the graph
builder = StateGraph(Agentstate)
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")
builder.add_node("llm",llm_node)
builder.add_node("file_writer",file_node)
builder.add_node("coder",coder_node)
builder.add_node("researcher",research_node)
builder.add_node("alpha_vantage",alpha_vantage_node)
builder.add_node("gdp_researcher",gdp_node)
builder.add_node("news_data",news_node)

# Set the configuration needed for the state
config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()

# Add edges from members to supervisor
for member in members:
    builder.add_edge(member, "supervisor")

# The supervisor populates the next field in the graph state to route to another node or finish
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("llm", "supervisor")


# Compile the graph
graph = builder.compile(checkpointer=memory)

# Try to draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
except Exception as e:
    print(f"Could not generate graph visualization: {e}")

# Main loop
def main_loop():
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit"]:
            print("GoodBye!")
            break
        
        try:
            for s in graph.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },
                config=config,
            ):
                print(s)
                print("-----")
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main_loop() 
