from langgraph.graph import StateGraph,END 
from parser import State 
from node import Supervisor, LLM, RAG 

def router(state:State):
    
    last_message=state["messages"][-1]
    
    if "food recipe" in last_message.lower():
        return "RAG Call"
    else:
        return "LLM Call"
    
def create_workflow():

    workflow = StateGraph(State)
    workflow.add_node("Supervisor" , Supervisor )
    workflow.add_node("LLM", LLM) 
    workflow.add_node("RAG", RAG)

    workflow.set_entry_point("Supervisor") 

    workflow.add_conditional_edges(
        "Supervisor",
        router,
        {
            "RAG Call": "RAG",
            "LLM Call": "LLM"
        }
    ) 

    workflow.add_edge("RAG" , END )
    workflow.add_edge("LLM" , END ) 

    return workflow 




