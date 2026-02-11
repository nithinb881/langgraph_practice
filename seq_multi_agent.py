import os
import time 
from datetime import datetime 
from typing import TypedDict 
from typing import Annotated, List 
from langgraph.graph import StateGraph,START,END 
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import HumanMessage,AIMessage , SystemMessage 
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq
from dotenv import load_dotenv  
load_dotenv() 


# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     timeout=None,
#     max_retries=2,
#     max_completion_tokens=100,
# )

# llm = ChatGroq(
#     model = "llama-3.3-70b-versatile",
#     max_retries= 3,
#     api_key= ""
# )

#define state  
class Graphstate(TypedDict):
    messages : Annotated[list, add_messages]

#build the graph 
builder = StateGraph(Graphstate) 

#utility Functions ( Using the utility function we are creating the various agents)
def create_node(state,system_prompt): 
    human_messages = [msg for msg in state["messages"] if isinstance
    (msg,HumanMessage)]
    ai_messages = [msg for msg in state["messages"] if isinstance
    (msg,AIMessage)]
    system_message = [SystemMessage(content=system_prompt)] 
    messages = system_message + human_messages + ai_messages 
    message = llm.invoke(messages)
    return {"messages":[message]} 

# Add nodes  

analyst = lambda state: create_node(state,""" 
You are a software requirement analyst. Review the provided information and generate the 
software development requirements that developer can understand and create a code from it.
Be precise and clear in your requirements                                                                       
""") 


architect = lambda state: create_node(state,""" 
You are an software architect who can design the stable system that can work in cloud environment 
Review the software requirements provided and create an architect document that will be used by 
developers, tester and designer to implement the system. Provide the architect only. 
""")

developer = lambda state: create_node(state,""" 
Your are an Full stack developer who can code in any language. Review the provide information
and write the code. Return only the code artifacts only.                                      
""")

reviewer = lambda state: create_node(state,""" 
Your an experience developer and code reviwer. You know the best design patterns for web applications 
that can run on the cloud and can do code review in any language. Review the provided code and suggest
the improvements. only focus on the provided code and suggest actionable items.                                                                          
""") 

tester = lambda state: create_node(state,""" 
You are an test automation expert who can create the test scripts in any language. Review the provided
user requirements and software requirments and write the test code to ensure good quailty of software.                                   
""") 

diagram_designer = lambda state: create_node(state, """
You are a Software Designer and can draw diagrams explaining any code. 
Review the provided code and create a Mermaid diagram explaining the code.""")

summary_writer = lambda state: create_node(state, """ 
You are an expert in creating technical documentation and can summarize complex documents into human-readable documents. 
Review the provided messages and create a meaningful summary. Retain all the source code generated and include it in the summary.""")

#Add nodes to graph 
builder.add_node("analyst",analyst) 
builder.add_node("architect",architect)
builder.add_node("developer",developer)
builder.add_node("reviewer",reviewer)
builder.add_node("tester",tester)
builder.add_node("diagram_designer",diagram_designer)
builder.add_node("summary_writer",summary_writer)



# enter and end points for graph 
builder.add_edge(START,"analyst")
builder.add_edge("analyst","architect")
builder.add_edge("architect","developer")
builder.add_edge("developer","reviewer")
builder.add_edge("reviewer","tester")
builder.add_edge("tester","diagram_designer")
builder.add_edge("diagram_designer","summary_writer")
builder.add_edge("summary_writer",END)





# complie and run the  
graph = builder.compile() 

#draw the graph 
try: 
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="multiagent_graph.png")
except Exception as e:
    print(f"Could not generate graph visualization: {e}")


# Create an main loop 
def main_loop():
    while True:
        user_input = input(">>") 
        if user_input.lower() in ["quit", "exit", "q"]:
            print("GoodBye!")
            break

        response = graph.invoke({"messages": [HumanMessage(content=user_input)]}) 
        print("Analyst:", response["messages"][-7].content)
        print("Architect:", response["messages"][-6].content)
        print("Developer:", response["messages"][-5].content)
        print("Reviewer:", response["messages"][-4].content)
        print("Tester:", response["messages"][-3].content)
        print("Diagram_designer:", response["messages"][-2].content)
        print("Summary_writer:", response["messages"][-1].content)

# run the main loop 
if __name__ == "__main__":
    main_loop()        
