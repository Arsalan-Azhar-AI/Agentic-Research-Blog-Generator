import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from langgraph.graph import StateGraph, END
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from src.state import AgentState
from src.node import (
    init,
    decompose_query,
    tavily_node,
    arxiv_node,
    wiki_node,
    reranker_node,
    generate_context,
    human_loop
)
from dotenv import load_dotenv
load_dotenv()

DB_URI = os.environ["DB_URI"]


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # On first use, create the necessary tables
    checkpointer.setup()
    # Now you can use the checkpointer in your LangGraph run
    # e.g., graph = builder.compile(checkpointer = checkpointer)

workflow = StateGraph(AgentState)

workflow.add_node("initial_state", init)
workflow.add_node("sub_query", decompose_query)
workflow.add_node("tavily_results", tavily_node)
workflow.add_node("arxiv_results", arxiv_node)
workflow.add_node("wiki_results", wiki_node)
workflow.add_node("reranker_results_node", reranker_node)
workflow.add_node("generate_context_node", generate_context)
workflow.add_node("human_loop", human_loop)

workflow.set_entry_point("initial_state")

workflow.add_edge("initial_state", "sub_query")
workflow.add_edge("sub_query", "tavily_results")
workflow.add_edge("sub_query", "arxiv_results")
workflow.add_edge("sub_query", "wiki_results")
workflow.add_edge("tavily_results", "reranker_results_node")
workflow.add_edge("arxiv_results", "reranker_results_node")
workflow.add_edge("wiki_results", "reranker_results_node")
workflow.add_edge("reranker_results_node", "generate_context_node")
workflow.add_edge("generate_context_node", "human_loop")
workflow.add_edge("human_loop", "generate_context_node")
workflow.add_edge("human_loop", END)


#checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

'''
question="What is AI?"
input_data = {
    "question": question,
    "queries": [],
    "combine_results": [],
    "reranker_results":[],
    "generate_context": "",
    "user_feedback": ""
}

thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}


result = graph.invoke(input_data, config=thread_config)
'''