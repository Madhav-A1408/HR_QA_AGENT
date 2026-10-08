from dotenv import load_dotenv
load_dotenv()

import json
import hashlib
import shutil
from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document       
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.types import Command

import config as cfg



# ---------------------------------------------------------------
# 1. Initializing LLM Model + Data (via config.py)
# ---------------------------------------------------------------

llm = init_chat_model("openai:gpt-5-nano")


# Data: policy documents loaded from JSON into a list the search tool reads.

with open(cfg.CORPUS_PATH, encoding = "utf-8") as f:
    CORPUS = json.load(f)


# ---------------------------------------------------------------
# 2. Vector store (built once, saved to disk, rebuilt if the JSON changes)
# ---------------------------------------------------------------

def build_vectorstore() -> Chroma:
    
    embeddings = OpenAIEmbeddings(model = cfg.EMBEDDING_MODEL)
    path = Path(cfg.CHROMA_PATH)
    hash_file = path.with_suffix(".hash")   
    current_hash = hashlib.md5(Path(cfg.CORPUS_PATH).read_bytes()).hexdigest()
    
    # Same idea as your RAG agent: reuse the saved index if it exists.
    # The hash also tells us if policy_corpus.json was edited since.
    
    if path.exists() and hash_file.exists() and hash_file.read_text() == current_hash:
        return Chroma(
            collection_name="policy_corpus",
            persist_directory=str(path),
            embedding_function= embeddings,
        )
        
    print("Building Policy Index")
    if path.exists():
        shutil.rmtree(path)
        
    docs = [
        Document(
             # this text is what gets turned into numbers and matched by meaning
             
             page_content=(
                f"{d['title']}. {d['section_title']}. "
                f"{' '.join(d['tags'])}. {d['content']}"
            ),
            # this is what we show back to the agent
            metadata={
                "title": d["title"],
                "category": d["category"],
                "source": d["source"],
                "content": d["content"],
            },
        )
        for d in CORPUS
    ]
    
    store = Chroma.from_documents(
        docs,
        embeddings,
        ids=[d["id"] for d in CORPUS],
        collection_name="policy_docs",
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},   # distance = 1 - cosine similarity
    )
    hash_file.write_text(current_hash)
    return store

vectorstore = build_vectorstore()
# ---------------------------------------------------------------
# 3. State
# ---------------------------------------------------------------
    
    # State = the shared notebook passed between all steps (router, LLM, tools).
    # Each step reads from it and writes updates back.
    #    messages                -> chat history (comes from MessagesState; new messages are appended)
    #    employee_id             -> who is asking
    #    classification_decision -> router's verdict: "answer" or "ignore"
    #    final_answer            -> reply for the employee; setting it ends the loop
    

class State(MessagesState):
    employee_id : str
    classification_decision : Literal["ignore","answer"]
    final_answer : str
    



# ---------------------------------------------------------------
# 4. Tools
# ---------------------------------------------------------------

@tool
def search_policy_docs(query:str,category:str = "") -> str:
    
    """Search the company policy documents. Use a short search phrase,
    e.g. 'casual leave entitlement'. Optionally narrow by category:
    leave, work-arrangements, payroll, benefits, conduct, performance, exit.
    Leave category empty if unsure."""

    results = vectorstore.similarity_search_with_score(
        query,
        k = cfg.TOP_K,
        filter={"category":category} if category else None,
    )
    
    