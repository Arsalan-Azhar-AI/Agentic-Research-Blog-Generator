import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import operator
from typing import List, Optional, Annotated, TypedDict
from pydantic import BaseModel, Field


class DecomposedQueries(BaseModel):
    """Decomposition of a complex query into simpler sub-questions."""
    questions: List[str] = Field(
        description="A list of self-contained sub-questions derived from the original query."
    )

class generate_structure(BaseModel):
  """Generate Blog output structure"""
  title: str = Field(description="The title of the generated blog")
  introduction: str = Field(descrition="The Introduction of the generated blog")
  body_content: str = Field(descrition="The main content of the generated blog")
  visuals_context: str = Field(descrition="The text of the blog image")
  conclusion: str = Field(descrition="The final summary of the blog")


class AgentState(TypedDict):
    """State for our agentic workflow"""
    question: str
    queries: List[str]
    #tavily_results: Optional[List[str]]
    #arxiv_results: Optional[List[str]]
    #wiki_results: Optional[List[str]]
    combine_results: Annotated[List[dict], operator.add]
    reranker_results: List[dict]
    generate_context: generate_structure
    user_feedback:str

