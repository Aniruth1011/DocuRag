from pydantic import BaseModel, Field  
from typing import List 

class RecepieOutputParser(BaseModel):

    ingredients : List[str] = Field(description="List of ingredients for the recipe")
    instructions : str = Field(description="Instructions for the recipe")  

class SuperVisorOutputParser(BaseModel):
    Topic:str=Field(description="selected topic")
    Reasoning:str=Field(description='Reasoning behind topic selection')

 