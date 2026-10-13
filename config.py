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
CHROMA_PATH = BASE_DIR / "chroma_data"


# ---------- Model settings ----------

MODEL_NAME = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------- Search settings ----------
TOP_K = 2                  # how many policies search_policy_docs returns at most
MAX_DISTANCE = 0.7         # cosine distance cutoff: lower = stricter (a starting guess, tune it)
DEBUG_SEARCH = True        # prints each search's distances to the console; turn off when tuned
# MAX_SEARCHES = 3           # after this many searches in one question, the agent must answer



# ---------- Agent settings ----------
RECURSION_LIMIT = 15            # safety cap on the LLM <-> tools loop
IGNORE_REPLY = "Sorry, I can only help with company policy and HR questions."


# ---------- Employee records, loaded from the JSON file ----------
with open(EMPLOYEES_PATH, encoding="utf-8") as f:
    EMPLOYEE_RECORDS = {str(e["EmpId"]): e for e in json.load(f)}
    
# converts it from a raw json to dict format.

