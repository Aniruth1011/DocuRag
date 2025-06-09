import os 
from dotenv import load_dotenv 
load_dotenv() 

from graph_workflow import create_workflow 

def load_env_variables():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") 
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") 
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")




     
