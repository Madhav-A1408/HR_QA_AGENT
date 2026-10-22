"""
evaluators.py  -  the GRADERS
=====================================================================
Think of the evaluation as an exam:
  - the goldens (manual + synthetic) are the ANSWER KEY
  - Sparc's reply is the STUDENT'S ANSWER
  - each function in this file is one EXAMINER who checks one thing

LangSmith runs every examiner on every test case. Each examiner receives
some of these three things (only the ones it asks for in its brackets):
  inputs            -> what the employee typed (the exam question)
  outputs           -> what Sparc actually did (comes from target() in run_eval.py)
  reference_outputs -> what Sparc SHOULD have done (the answer key entry)

Every examiner hands back a small report card:
  {"key": name of the check, "score": True (pass) / False (fail),
   "comment": a short reason, so you know WHY it passed or failed}

Examiners 1-4 are plain Python checks: free, instant, same result every time.
Examiners 5-6 ask an LLM (the "judge") to read the reply, for things only
a reader can decide, like "is this answer correct?".
=====================================================================
"""

import json
import re
import sys
from pathlib import Path

# BaseModel, Field: define the fixed shape the judge LLM must reply.
# init_chat_model: creates the judge LLM.

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import config as cfg  

# Make a list of every section number that REALLY exists in the handbook
# (e.g. {"1.1", "1.2", "2.1", ...}). Examiner 3 uses it to catch
# made-up section numbers like "Section 9.9".
with open(cfg.CORPUS_PATH,encoding = "utf-8") as f:
        REAL_SECTIONS = {d["section"] for d in json.load(f)}
        
    
    

# =====================================================================
# EXAMINER 1: TRIAGE
# Question it answers: "Did the router make the right first decision?"
# Example: "Tell me a joke" should be "ignore"; "When is salary credited?"
# should be "answer".
# =====================================================================

def triage_eval(outputs:dict,reference_outputs:dict) -> dict:
    got = outputs["classification_decision"]  # what the router actually decided
    want = reference_outputs["classification"] # what the answer key says it should decide
    return {
        "key": "triage_correct",
        "score": got==want,                 # pass only if they are exactly the same
        "comment": f"expected = {want} got = {got}" 
    }
    
# =====================================================================
# EXAMINER 2: TOOLS
# Question it answers: "Did Sparc use the right tools, and avoid the wrong ones?"
# Example: for "When is salary credited?" it MUST call search_policy_docs,
# and must NOT call Question (asking for an ID is pointless there).
# ====================================================================
def tools_eval(outputs:dict,reference_outputs:dict) -> dict:
    # Every tool Sparc called during this run. A set() removes repeats,
    # so calling a tool (eg :search_policy_docs) twice still counts as "called". 
    called = set(outputs["tool_calls"])
    
    # Tools that SHOULD have been called but weren't
    missing = []
    for tool_name in reference_outputs["expected_tools"] :
        if tool_name not in called:
            missing.append(tool_name)   
            
    # Tools that should NOT have been called but were
    forbidden = []
    for tool_name in reference_outputs["expected_tools"]:
        if tool_name in called:
            forbidden.append(tool_name)
            
    
    # Pass only if nothing is missing AND nothing forbidden was used
    return {
        "key": "tools_correct",
        "score": not missing and not forbidden,                 # pass only if they are exactly the same
        "comment": f"called = {sorted(called)} missing = {missing},forbidden_used = {forbidden}" 
        }
    
    
    
# =====================================================================
# EXAMINER 3: CITATIONS
# Question it answers: "Did Sparc show where its facts came from, correctly?"
# Checks three things in the reply text:
#   a) every expected section is mentioned, e.g. "Section 2.1"
#   b) "Employee Record" is mentioned when the employee's own data was used
#   c) no fake section numbers that don't exist in the handbook
# =====================================================================
def citations_eval(outputs:dict,reference_outputs:dict) -> dict:
    answer = outputs["final_answer"]
    problems = []  # collect every problem found; empty list = all good
    
    # a) Is each expected section cited?
    for i in reference_outputs["expected_sections"]:
        if f"Section {i}" not in answer:
            problems.append(f"missing Section {i} ")
    
    # b) If the answer used the employee's own data, is that source cited?
    
    if reference_outputs["needs_record"] and "Employee Record not in abswer":
        problems.append("Missing ' Employee Record '")
    
    # c) Find every "Section X.Y" in the reply and check each one is real.
    #    re.findall pulls out just the numbers, e.g. "Section 2.1" -> "2.1"
    
    for i in re.findall(r"Section (\d+\.\d+)", answer):
            if i not in REAL_SECTIONS:
                problems.append(f"Invented Section {i} ")
                
    # Pass only if no problems were found           
    return {
            "key": "citation_correct",
            "score": not problems,                 
            "comment": "; ".join(problems) or "ok"
        }