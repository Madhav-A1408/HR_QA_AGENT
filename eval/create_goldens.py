"""
The full answer key for Sparc = hand-written cases + synthetic cases.

  1. Hand-written cases come from goldens/manual_goldens.json
  2. Synthetic general-policy cases come from goldens/  synthetic_goldens.json (if it exists)

Each case ends up in the shape LangSmith expects:
  {"inputs": {...what the employee types...}, "outputs": {...the reference answer...}}

Fields in "outputs":
  classification    -> "answer" or "ignore" (what triage should decide)
  expected_tools    -> tools that MUST be called
  forbidden_tools   -> tools that must NOT be called
  expected_sections -> handbook sections the reply must cite ("Section X.Y")
  needs_record      -> reply must cite "Employee Record"
  criteria          -> what a good reply says (graded by an LLM judge)
  must_not_contain  -> text that must never appear (e.g. another person's data)

run_eval.py imports DATASET_NAME and EXAMPLES from here.

"""

import json
from pathlib import Path

DATASET_NAME = "Evaluation Dataset"

GOLDENS_DIR = Path(__file__).parent / "goldens"
MANUAL_PATH = GOLDENS_DIR / "manual_goldens.json"
SYNTH_PATH = GOLDENS_DIR / "synthetic_goldens.json"




# This function takes a short description of a test and returns the full LangSmith shape.

def case(question,classification = "answer", tools = (), forbidden = (), sections = (), record = False, 
         criteria = "",must_not_contain = (),history = None):
    
     """Turn one short case into the full LangSmith shape, filling in defaults."""
     
     return {
        "inputs" : {"question":question, "history": history or []},
        "outputs" : {
            "classification":classification,
            "expected_tools": list(tools),
            "form=bidden_tools": list(forbidden),
            "expected_sections": list(sections),
            "needs_record": record,
            "criteria": criteria,
            "must_not_contain": list(must_not_contain),
            
        },
     }
     
# ---------- 1. Hand-written cases (from manual_cases.json) ----------
# Each JSON entry's keys match case()'s parameter names, so **entry fills them in.
# Any field left out of an entry gets the default shown in case() above.
     
GOLDENS = []

for i in json.loads(MANUAL_PATH.read_text(encoding="utf-8")):
    GOLDENS.append(case(**i))
    
        # #MANUAL_PATH.read_text(...) reads the file as text.
        # json.loads(...) converts that text into a Python list of dictionaries.
        # The loop takes each entry (one dictionary per test).
        # case(**entry) unpacks the dictionary into keyword arguments.



# ---------- 2. Synthetic cases (from synthetic_goldens.json) ----------
# No forbidden tools here: a synthetic "Can I...?" question about a policy
# with conditions makes asking for the ID corresct, not a failure.


if SYNTH_PATH.exists():
    for s in json.loads(SYNTH_PATH.read_text(encoding = "utf-8")):
        GOLDENS.append(case(
            s["question"],
            tools=["search_policy_docs"],
            sections = [s["section"]],
            criteria = f"Agrees with this reference answer on all facts and numbers: {s['reference_answer']}",
            
        ))