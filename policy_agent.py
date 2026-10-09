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

    # One call does three jobs inside Chroma:

        # Turns the query into numbers using the embedding model (this is the one tiny OpenAI call per search).
        # Compares those numbers with every saved section’s numbers.
        # Returns the closest ones, each paired with a distance score.
    
    
    if category:
        category_filter = {"category":category}
    else: 
        category_filter = None    
    
    results = vectorstore.similarity_search_with_score(
        query,
        k = cfg.TOP_K,
        filter= category_filter,
    )
    
    # Only runs when DEBUG_SEARCH = True. It prints what the AI searched for, then each result’s distance 
    # (3 decimal places) and title. It runs before the cutoff, so you can see the matches that are about to 
    # be dropped.
    if cfg.DEBUG_SEARCH:
        print(f"   [search] '{query}' (category = ''{category})")
        for doc, score in results:
            print(f"   {score:.3f} '{doc.metadata['title']}")

        
    # keep only close matches (lower distance = closer)
    # This is a list comprehension, which is a one-line loop. In plain words: “go through every result, 
    # and keep it only if its distance is at or below MAX_DISTANCE.”
   
    good= []
   
    for doc, score in results:
                if score <= cfg.MAX_DISTANCE:
                    good.append(doc)
                    
                if not good:
                    return "No Matching Policy Found!"
                
    # Turn each matched policy into one text block: [Title] content (Source: ...)
    # Values come from metadata (the clean original text), not the embedded page_content.
    # The source is included so the agent can cite it, as AGENT_PROMPT rule 3 requires.
    # Blocks are joined with a blank line and returned to the agent as a ToolMessage.
        
    lines = []
    for i in good:
        lines.append(
            f"[{doc.metadata['title']}] {doc.metadata['content']} "
            f"(Source: {doc.metadata['source']})"
        )
    return "\n\n".join(lines) 



@tool
def check_employee_record(field:str, config:RunnableConfig, employee_id: str = "") -> str:

    """Look up ONE field from the asking employee's own record.
    If you do not know the employee's ID yet, do not guess: call Question to
    ask for it (employee IDs are numbers like 384), then pass it as employee_id.
    Valid fields: Employee Name, Dept, Work Location, Designation, Level,
    Gender, Employment Type, Employment Status, Date of Joining,
    Reporting Manager, Casual Leave Balance (Days),
    Sick Leave Balance (Days), Earned Leave Balance (Days).
    """
    supplied_id = config["configurable"].get("employee_id")

    if supplied_id:
        employee_id = supplied_id
    
    if not employee_id:
        return "Employee ID not known. Call Question tool to ask the employee for thier Employee ID"     
    
    record = cfg.EMPLOYEE_RECORDS.get(str(employee_id).strip(),{})
    
    if not record:
        return(
            f"No employee found with ID '{employee_id}' . "
            "Call Question tool to ask the employees to re-check their Employee ID."
        ) 
        
    if field not in record:
        return f"No field '{field}'. Valid fields: {', '.join(record)}"
    
    return str(record[field])


@tool("Question")
def question(content:str) -> str:
    """
    Ask the employee ONE short clarifying question when you cannot answer
    without more information (e.g. 'Which type of leave do you mean?' or
    'What is your employee ID?').
    
    """
    
    return "Question sent to employee." 

       
                    
    
