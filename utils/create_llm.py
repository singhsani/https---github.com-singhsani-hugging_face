from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

#AIzaSyD9TFYUbHo2_QxLjP_V2YaNXsrnLLnoQpA
def create_google_llm():

    load_dotenv("D:/python/keyEnv.env")
    SECRETE_KEY =os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        api_key=SECRETE_KEY,   # directly paste key
        temperature=0
    )
    return llm