import os 
from dotenv import load_dotenv 
from  parser import RecepieOutputParser 
load_dotenv() 

from graph_workflow import create_workflow 

def load_env_variables():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") 
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") 
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

app = create_workflow().compile()
# state={"messages":]} 
# state={"messages":["I have mustard, tomatoes, oil , ghee and onions , carrots . What all different food items can i make with ?"]} 


def ask_ai(message): 

    state = {"messages" : [message]} 

    result = app.invoke(state) 

    print("Result:", result)

    if isinstance(result["messages"][-1] , RecepieOutputParser):
        result_inrgredients = result["messages"][-1].ingredients 
        result_instructions = result["messages"][-1].instructions 
        result = f"Ingredients: {result_inrgredients}\nInstructions: {result_instructions}" 
    else:
        
        result = result["messages"][-1].content if "messages" in result and result["messages"] else "No response from AI."
    return result 




     
