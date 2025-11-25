import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.tools import TavilySearchResults, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

tavily_tool = TavilySearchResults()


api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper)