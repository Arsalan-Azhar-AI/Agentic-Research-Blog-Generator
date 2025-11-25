import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import operator
from typing import List, Dict, Any, Optional, Annotated

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, interrupt
from langgraph.graph import END

from src.state import DecomposedQueries, AgentState,generate_structure
from src.llm_setup import llm
from src.embeddings_setup import embedding
from src.tools import tavily_tool,arxiv_tool,wiki_tool


def init(inputs):
  return {"question":inputs["question"]}

def decompose_query(inputs):
  question = inputs["question"]
  structured_llm = llm.with_structured_output(DecomposedQueries)
  response = structured_llm.invoke(question).questions
  return {"queries": response}


def _safe_run(tool, query: str, source: str) -> dict:
    try:
        out = tool.run(query)
        return {"source": source, "query": query, "content": out}
    except Exception as e:
        return {"source": source, "query": query, "error": str(e)}


def tavily_node(inputs):
    queries = inputs.get("queries")
    results = [_safe_run(tavily_tool, q, "tavily") for q in queries]
    return {"combine_results": results}

def arxiv_node(inputs):
    queries = inputs.get("queries")
    results = [_safe_run(arxiv_tool, q, "arxiv") for q in queries]
    return {"combine_results": results}

def wiki_node(inputs):
    queries = inputs.get("queries")
    results = [_safe_run(wiki_tool, q, "wiki") for q in queries]
    return {"combine_results": results}



def semantic_search(documents):
  vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_metadata={"hnsw:space": "cosine"}
    )
  vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
  return vector_retriever
def keyword_search(documents):
  bm25_retriever = BM25Retriever.from_documents(documents)
  bm25_retriever.k = 15
  return bm25_retriever

def hybrid_search(documents,query):
  vector_retriever=semantic_search(documents)
  bm25_retriever=keyword_search(documents)
  hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
    )
  retrieved_docs = hybrid_retriever.invoke(query)
  return retrieved_docs

reranker = CohereRerank(model="rerank-english-v3.0", top_n=10)

def reranker_node(inputs: AgentState) -> AgentState:
    query = inputs['question']
    chunks = inputs['combine_results']  # [{}, {}, {}]

    ###############################################
    # 1. Convert each chunk to a Document
    ###############################################
    documents = []

    for chunk in chunks:
        content = chunk.get("content", "")

        # Handle Tavily case (list of dicts)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    text_parts.append(f"Title: {title}\nSnippet: {snippet}")
                else:
                    text_parts.append(str(item))
            content = "\n\n".join(text_parts)

        # Create LangChain Document
        doc = Document(
            page_content=str(content),
            metadata={
                "source": chunk.get("source", ""),
                "query": chunk.get("query", "")
            }
        )
        documents.append(doc)

    ###############################################
    # 2. Run hybrid search (semantic + keyword)
    ###############################################
    retrieved_docs = hybrid_search(documents, query)

    ###############################################
    # 3. Apply Cohere reranker
    ###############################################
    reranked_docs = reranker.compress_documents(retrieved_docs, query)

    ###############################################
    # 4. Convert back to same structure as combine_results
    ###############################################
    final_results = []
    for doc in reranked_docs:
        final_results.append({
            "source": doc.metadata.get("source", ""),
            "query": doc.metadata.get("query", ""),
            "content": doc.page_content
        })

    ###############################################
    # 5. Return updated state
    ###############################################
    inputs["reranker_results"] = final_results
    return inputs

prompt = ChatPromptTemplate.from_messages([
  ("system", """You are an expert Blog generator. Generate the Blog based only on the provided context.\nif user pass feedback generate blog based on context and feedback. \n     If you don't know the answer, say you don't know. Be detailed and specific when possible."""),
  ("human", """Question: {question}\nContext:\n{context}\nUser_feedback:\n{user_feedback}""")
])


def generate_context(inputs:AgentState)->AgentState:
    combine_results = inputs.get("reranker_results")
    user_feedback=inputs.get("user_feedback")
    parts = []
    for item in combine_results:
        source = item.get("source", "unknown")
        query = item.get("query", "")
        if "error" in item:
            parts.append(f"Source: {source}\nQuery: {query}\nError: {item.get('error')}")
        else:
            content = item.get("content", "")
            if isinstance(content, (dict, list)):
                content = repr(content)
            parts.append(f"Source: {source}\nQuery: {query}\nContent: {content}")

    context = "\n\n---\n\n".join(parts)
    rag_chain = prompt | llm.with_structured_output(generate_structure)

    inputs["generate_context"] = rag_chain.invoke({
      "question": inputs["question"],
      "context": context,
      "user_feedback":user_feedback
  })
    return Command(
        update={"generate_context": inputs["generate_context"]},
        goto="human_loop"   # 👉 graph will automatically jump to human_loop node
    )


def human_loop(inputs: AgentState) -> AgentState:
    print("\n✅ === Entering Blog Approval State ===")
    print("Generated Blog\n")
    print(inputs.get("generate_context"))

    # Ask the user whether they want to finish or provide feedback
    approval = interrupt("Provide feedback or type 'done' to finish")

    if isinstance(approval, str) and approval.strip().lower() == "done":
        # Mark finalised and end
        return Command(
            update={"user_feedback": inputs.get("user_feedback", "") + "Finalised"},
            goto=END
        )

    # Otherwise capture the feedback (explicitly into a variable)
    feedback_text = interrupt("Provide feedback")

    # Append safely (use get to avoid KeyError if user_feedback missing)
    new_feedback = inputs.get("user_feedback", "") + feedback_text

    # Return a Command that updates state and jumps back to generator
    return Command(
        update={"user_feedback": new_feedback},
        goto="generate_context"   # loops back to regenerate using updated state
    )
