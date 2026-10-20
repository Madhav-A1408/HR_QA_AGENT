"""
Generates general policy test cases from policy_corpus.json using DeepEval's Synthesizer.
Called automatically by run_eval.py when synthetic_goldens.json doesn't exist.

"""


import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#Reading this from the inside out:
# __file__ is the path of this script, e.g. project/eval/synthesize_goldens.py.
# .resolve() turns it into a full absolute path.
# .parent goes up one folder (to project/eval).
# .parent again goes up another (to project).
# So ROOT is the project’s main folder. The script lives in the eval/ subfolder, so we go up two levels to reach 
# the project root, where config.py and .env sit.


sys.path.insert(0,str(ROOT))

#Because this script is inside eval/, Python wouldn’t find config.py in the parent folder. 
# This line adds the project root to the front of the list (position 0), so import config works.


from dotenv import load_dotenv
load_dotenv(ROOT /  ".env")

#This reads your .env file and loads OPENAI_API_KEY into the environment. 
# It points to ROOT / ".env" explicitly, so it works no matter which folder
# you launch the script from.

import config as cfg

OUT_PATH = Path(__file__).parent / "synthetic_goldens.json"  

# Where the finished test cases will be saved.

GOLDENS_PER_POLICY = 2 

#  21 policies x 2 = about 40 questions

SYNTH_MODEL = cfg.SYNTH_MODEL




def synthesize() -> int:
    """Generate synthetic cases and save them to OUT_PATH. Returns how many were saved."""
    
    from deepeval.synthesizer import synthesizer
    
    with open(cfg.CORPUS_PATH,encoding = "utf-8") as f:
        corpus = json.load(f)
        
    contexts = [[d["content"]] for d in corpus]
    
    # For each policy d, it takes the content text and wraps it in its own list.
    # DeepEval’s generate_goldens_from_contexts expects a list of contexts, and every context has exactly one policy. 
    # That is what makes each question trace back to one specific policy,
    
    
    policy_by_content = {d["content"]: d for d in corpus}
    
    # This makes a dictionary where the key is the policy text and the value is the whole policy record:
    
    synthesizer = synthesizer(model = SYNTH_MODEL)
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts = contexts,
        include_expected_outputs = True,
        max_goldens_per_output = GOLDENS_PER_POLICY,
        
    )
    
    #This is the main step. It sends each context to gpt-4o and gets back questions. The three settings:

    # contexts=contexts: the policies to read.
    # include_expected_output=True: also write the correct answer for each question. 
    # Without this you’d get questions only. This answer becomes your reference_answer.
    # max_goldens_per_context=2: at most 2 questions per policy.

   
    #  Tag each question with the section it was generated from
    test_cases = []
    for g in goldens:
        policy = policy_by_content.get(g.context[0])
        
        if policy is None:
            continue
        
        test_cases.append({
            "question" : g.input,
            "expected_answer" : g.expected_output,
            "section" : policy["section"],
            "policy_title" : policy["title"],
            
        })
        
    OUT_PATH.write_text(json.dumps(test_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(test_cases)
   
    
    





