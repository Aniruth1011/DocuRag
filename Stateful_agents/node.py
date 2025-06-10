from typing import TypedDict , Annotated 
from langchain_core.tools import tool 
from langchain_tavily import TavilySearch 
from langchain_core.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.vectorstores import FAISS
from parser import RecepieOutputParser  , SuperVisorOutputParser 
import operator 

import os 
from dotenv import load_dotenv 
load_dotenv() 
LLM_TYPE = os.getenv("LLM_TYPE")
LLM_MODEL = os.getenv("LLM_MODEL") 

class State(TypedDict):
    messages : Annotated[list[str], operator.add]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool 
def WebCrawlerTool(state:State):


    """WebCrawlerTool is a tool that performs a web search using TavilySearch and returns the results."""  

    print("inside web crawler tool")

    tool = TavilySearch(
    max_results=25,
    topic="general",
    include_answers = True, 
    include_images=True,
    search_depth = "advanced" ,
    include_related_searches = True, 
    include_snippets = True, 
    include_news = True,
    include_videos = True,
    include_websites = True, 
    include_wikipedia = True,
    include_books = True,
    include_products = True,
    include_questions = True,
    include_people = True,
    include_places = True,
    include_events = True,
    include_software = True,
    include_apps = True,
    include_scholar = True,
    include_jobs = True,
    include_facts = True,
    include_definitions = True,
    include_translations = True,
    include_summaries = True,
    include_lyrics = True,
    include_quotes = True,

    )

    search_query =state["messages"][0]

    #print("inside tool " , search_query)
    search_snippets = tool.invoke(search_query) 

    if LLM_TYPE == "GROQ":
        llm = ChatGroq(model=LLM_MODEL ,  temperature=0.1) 
    elif LLM_TYPE == "google":
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL , temperature=0.1)   

    prompt = PromptTemplate(
        input_variables=["search_snippets"],
        template="""
    You are a smart assistant tasked with reading multiple short text outputs from search results.

    Each snippet contains useful information about a topic.

    Your job is to write a well-detailed, informative description that brings together the main points from all the snippets.

    Here are the snippets:
    {search_snippets}

    Write a clear, well-detailed description of what these results are saying:
    """
    )

    print("Search Snippets: ", search_snippets)
    results= llm.invoke(prompt.invoke({"search_snippets": search_snippets})) 

    print("results" , results)
    return  results

def LLM(state:State):

    if LLM_TYPE == "GROQ":
        llm = ChatGroq(model=LLM_MODEL ,  temperature=0.1) 
    elif LLM_TYPE == "google":
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL , temperature=0.1)   

    query = (
    "You are given a question that may require either general knowledge or recent, up-to-date information.\n\n"
    "If you can confidently answer the question using only the knowledge you've been trained on, do so.\n"
    "However, if the question involves current events, recent developments, newly released information, or any topic likely "
    "to have changed since your training data cutoff, you must use the WebCrawler tool to perform a live search before answering.\n\n"
    "Do not ask whether to use the WebCrawler. Make the decision yourself based on the nature of the question.\n"
    "If you need it, use the WebCrawler automatically. If not, answer using your internal knowledge.\n\n"
    "**Examples:**\n"
    "- Use trained knowledge for: 'Explain how photosynthesis works' or 'What is the theory of relativity?'\n"
    "- Use the WebCrawler for: 'What is the latest news on Nvidia’s stock?', 'Who won the 2024 elections?', "
    "or 'Is there a new iPhone released this month?'\n\n"
    "Here is the question:\n\n" + state["messages"][0]
    )

    tools_list = [WebCrawlerTool]
    llm = llm.bind_tools(tools_list)


    #print("Query to LLM" , query)
    result = llm.invoke(query) 

    if isinstance(result, AIMessage) and result.tool_calls:
        tool_call = result.tool_calls[0]  
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        #print(tool_call , tool_name , tool_args)
        
        if tool_name == "WebCrawlerTool":
            print("Tool input" , tool_args["state"])
            print("State" , State(tool_args["state"]))
            tool_output = WebCrawlerTool({"state" : tool_args["state"]})

            #print("&&&&&&&&&&" , tool_output)
            tool_result_msg = ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            )

            final_response = llm.invoke([result, tool_result_msg])
            return {"messages": [final_response]}

    return {"messages": [ result]}


def RAG(state:State):

    if LLM_TYPE == "GROQ":
        llm = ChatGroq(model=LLM_MODEL, max_tokens=1000, temperature=0.1) 
    elif LLM_TYPE == "google":
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL , temperature=0.1)   

    parser = PydanticOutputParser(pydantic_object=RecepieOutputParser)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
        "Using the following context, answer the question.\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Format your answer as follows:\n{format_instructions}\n\n"
        "Answer:"
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},

        
    )

    query = state["messages"][0] 

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.load_local(
    "faiss_recepie_ivf_index", embeddings, allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever()

    #retriever = vector_store.as_retriever()

    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )


    result = rag_chain.invoke(query) 

    return {"messages": [ result]}

def Supervisor(state: State):
    from langchain.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    question = state["messages"][-1]

    #print("Supervisor received question:", question)

    parser = PydanticOutputParser(pydantic_object=SuperVisorOutputParser)
    
    if LLM_TYPE == "GROQ":
        llm = ChatGroq(model=LLM_MODEL, max_tokens=1000, temperature=0.1)
    elif LLM_TYPE == "google":
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL , temperature=0.1)   


    template = """
    You are an intelligent assistant that classifies user queries into two categories: "Food Recipe" or "General".
    
    Your job is to read the user’s query, understand the context, and decide which category it belongs to, based on the following definitions:

    - **Food Recipe**: Questions asking about how to cook something, steps to prepare a meal, required ingredients, cooking tools or methods, food combinations (e.g., "What can I cook with tomatoes and cheese?"), or anything related to meal preparation or kitchen processes.

    - **General**: Any query that is not explicitly about food, cooking, or recipes. This includes current events, historical facts, science, technology, entertainment, personal opinions, advice, or anything not related to recipe preparation.

    --- 
    Instructions:
    - The "Topic" field should be either "Food Recipe" or "General".
    - Your reasoning should be 1–3 sentences explaining why the query fits the chosen category.
    - Do not add any commentary or extra explanation outside the JSON format.
    
    --- 
    User Query:
    {question}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    #print("Running Supervisor with question:", question) 
    # print("Prompt : " ,  prompt.template)
    
    response = chain.invoke({"question": question})

    #print("Parsed response:", response)

    return {"messages": [response.Topic], "topic_reasoning": response.Reasoning}

