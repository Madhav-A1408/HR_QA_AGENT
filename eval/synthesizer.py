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

