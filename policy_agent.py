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



@tool("Done")
def done(answer:str) -> str:
    """
    Call this when you are finished. Put the final reply for the employee
    in 'answer'. If you cannot answer safely, the answer should tell them
    to contact HR.
    
    """
    
    return "Answer sent to Employee." 


# A list of all four tools for the AI, plus a lookup dictionary keyed by tool name.
tools = [search_policy_docs,check_employee_record,question,done]

tools_by_name = {}
for t in tools:
    tools_by_name[t.name] = t



# ---------------------------------------------------------------
# 6. TRIAGE ROUTER 
#    WHAT: decides if a message is an HR/policy question ("answer")
#          or off-topic ("ignore").
#    HOW: one LLM call that must reply in a fixed shape (RouterSchema),
#         so the answer is always exactly "answer" or "ignore".
# ---------------------------------------------------------------

# Setting the shape of router's reply.
class RouterSchema(BaseModel):
    reasoning:str = Field(description="One or two sentences of Reasoning")
    classification:Literal["ignore", "answer"] = Field(description="'ignore' if NOT a company policy/HR question,"
                                                       "'answer' if it is ")   
    
        
# Wrapping the LLM with the RouterSchema shape so its reply always matches this format.

router_llm =llm.with_structured_output(RouterSchema)

# Triage Router instructions
TRIAGE_PROMPT = """You are the first filter for a company Policy/HR assistant.
- 'answer': questions about company policy, leave, benefits, work-from-home,
  notice periods, payroll, conduct/harassment, or the employee's own HR data
  (like leave balance).
- 'ignore': anything else (jokes, weather, coding help, random chat).

Examples:
- "Can I quit tomorrow?" -> answer
- "My manager harassed me, what do I do?" -> answer
- "How many sick days do I have left?" -> answer
- "What's a good laptop for coding?" -> ignore
- "Tell me a joke" -> ignore

If the latest message answers a clarifying question the assistant just asked
(for example an employee ID or 'confirmed'), choose 'answer'.
If unsure, choose 'answer'. It is safer to let the response agent say
"contact HR" than to turn away a real HR question."""


def triage_router(state:State) -> Command[Literal["response_agent","__end__"]]:
     # Step 1: read the last few messages (not just the latest), so a short reply
    
    
    #Builds a mini script from the last 4 messages. It keeps only human and AI messages that have real text,
    # which drops tool results and the AI’s silent tool calls.
    recent_messages = state["messages"][-4:]
    lines = []
    for m in recent_messages:
        if m.type in ("human","ai") and m.content:
            lines.append(f"{m.type}:{m.content}")
            
    transcript = "\n".join(lines)
    
    
    
    #  Sends the instruction sheet and the mini script to the classifier.    
    result = router_llm.invoke(
        [SystemMessage(content = TRIAGE_PROMPT), HumanMessage(content = transcript)]
        
    )
    
    
    # answer: record the verdict, clear final_answer (so a leftover can’t end the loop too early), 
    # and go to the agent.
    # ignore: record the verdict, set the polite canned reply from config, and end.
    
    if result.classification == "answer":
        return Command(
            goto="response_agent",
            update = {"classification_decision":"answer","final_answer":""},
        )
        
    return Command(
        goto=END,
        update = {"classification_decision":"ignore","final_answer":cfg.IGNORE_REPLY},
    )
    


# ---------------------------------------------------------------
# 7. RESPONSE AGENT 
#    WHAT: loops "LLM picks a tool -> tool runs -> LLM sees the result"
#          until it calls Done or Question.
# ---------------------------------------------------------------

# The agent's instructions. Rules 1-7 control how it behaves.
AGENT_PROMPT = """You are a Policy/HR assistant for employees.
Tools:
- search_policy_docs: find company policy text.
- check_employee_record: look up the asking employee's own data (designation,
  employment status, gender, joining date, manager, leave balances, etc.).
  It needs the employee's ID.
- Question: ask the employee ONE short clarifying question.
- Done: send the final reply.

Rules:
1. Use search_policy_docs for any policy fact. Never answer policy facts from memory.
2. Use check_employee_record only for the asker's own data.
3. Base answers only on tool results. End every policy answer with the source
   exactly as given, in this form: (Source: HR Policy Handbook, Section 7.1)
4. If the tools don't give a clear answer, or the question needs a judgment or
   exception, call Done with a reply telling the employee to contact HR.
5. If the answer depends on a detail you don't have (for example which leave
   type), first try check_employee_record. If it still isn't known, call
   Question with ONE short question. Never guess.
6. check_employee_record needs the employee's ID. If you don't know it, or the
   tool says it is not known or not found, call Question to ask for it. If the
   employee already gave their ID earlier in this conversation, reuse it and
   do not ask again. Pass it as employee_id. Never invent an ID.
7. Always finish by calling Done or Question."""   


#Hands the four tools to the AI. tool_choice="required" means it must call a tool every turn and can’t just chat. 
# So the only way to finish is Done or Question.    

llm_with_tools = llm.bind_tools(tools,tool_choice="required")


# --- Node: llm_call -------------------------------------------------
# WHAT: shows the LLM the instructions + conversation so far, and lets it
#       choose the next tool.

def llm_call(state:State):
    response = llm_with_tools.invoke(
        [SystemMessage(content=AGENT_PROMPT)+state["messages"]]
    )
    return {"messages":[response]}




# --- Node: tool_node ------------------------------------------------
# WHAT: runs the tool(s) the LLM just chose and records the results.
# SPECIAL: if the tool was Done or Question, its text becomes final_answer
#          (the reply to the employee).


def tool_node(state:State, config : RunnableConfig):
    results = []
    final_answer = None
    
    last_message = state["messages"][-1] # the LLM's latest message holds its tool choice(s)
    for call in last_message.tool_calls:
        tools_to_run = tools_by_name[call["name"]]
        observation = tools_to_run[call["args"],config]  # actually run the tool
        results.append(
            ToolMessage(content = observation, tool_call_id =call["id"], name = call["name"] )
        )
        if call["name"] == "Done":
            final_answer = call["args"]["answer"]
        elif call["name"] == "Question":
            final_answer = call["args"]["answer"]
    
    update = {"messages":results}
    if final_answer:
        update ["final_answer"] = final_answer  
    return update



# --- Decision: should_continue --------------------------------------
# WHAT: after each tool run, decide: finish, or loop back to the LLM?

def should_continue(state:State) -> Literal["llm_call","__end__"]:
    if state.get("final_answer"): # Done or Question was called -> the reply is ready
        return END
    return "llm_call"  # otherwise let the LLM see the tool result and continue




        
            
    
    
     