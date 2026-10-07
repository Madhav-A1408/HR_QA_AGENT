"""
Single shared config for HR_QA_Agent.
Keeps file paths, model definition, agent settings etc in one place .

"""
import json
from pathlib import Path



# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "policy_corpus.json"
EMPLOYEES_PATH = DATA_DIR / "employee_records.json"


# ---------- Model settings ----------

MODEL_NAME = "gpt-5-nano"


# ---------- Agent settings ----------
TOP_K = 2                       # how many policies search_policy_docs returns
RECURSION_LIMIT = 15            # safety cap on the LLM <-> tools loop
DEFAULT_EMPLOYEE_ID = "E102"    # only a typing shortcut for chat.py's prompt
IGNORE_REPLY = "Sorry, I can only help with company policy and HR questions."


# ---------- Employee records, loaded from the JSON file ----------
with open(EMPLOYEES_PATH, encoding="utf-8") as f:
    EMPLOYEE_RECORDS = {str(e["EmpId"]): e for e in json.load(f)}
    
# converts it from a raw json to dict format.

