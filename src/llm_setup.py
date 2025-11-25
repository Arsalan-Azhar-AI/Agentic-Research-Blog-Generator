import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0
)
