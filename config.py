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



# ---------- Chatbot identity ----------
BOT_NAME = "Sparc"
COMPANY_NAME = "Experion Technologies"



# ---------- Model settings ----------

MODEL_NAME = "anthropic:claude-haiku-5-5"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------- Evaluation settings ----------
# Models used only by the eval/ scripts (not by Sparc itself).
# Keep them different from MODEL_NAME so the agent isn't grading or writing its own exam.
SYNTH_MODEL = "gpt-4o"            # writes synthetic test questions (DeepEval format: plain name)
JUDGE_MODEL = "anthropic:claude-sonnet-5-5"   # grades Sparc's answers (LangChain format: "provider:model")




# ---------- Search settings ----------
TOP_K = 2                  # how many policies search_policy_docs returns at most
MAX_DISTANCE = 0.7         # cosine distance cutoff: lower = stricter (a starting guess, tune it)
DEBUG_SEARCH = True        # prints each search's distances to the console; turn off when tuned
# Words in policy text that mean "this depends on who you are".
# If an employee asks about themselves and the policy contains one of these,
# the agent must check their record before answering.
CONDITION_WORDS = ["probation", "permanent", "intern", "full-time", "female", "male"]
# Which record field decides each condition word.
# If the policy found mentions the word, that field must be read before answering.
REQUIRED_FIELDS = {
    "female": "Gender",
    "male": "Gender",
    "probation": "Employment Status",
    "permanent": "Employment Status",
    "intern": "Employment Type",
    "full-time": "Employment Type",
}

RELATIVE_GAP = 0.15   # keep a result only if it's within this distance of the best match


# ---------- Agent settings ----------
RECURSION_LIMIT = 15            # safety cap on the LLM <-> tools loop

IGNORE_REPLIES = [
            "Sorry, I can only help with HR-related queries.",
            "That's outside what I can help with. Feel free to ask me any HR-related question.",
            "I'm only able to answer HR-related queries. Is there anything HR-related I can help with?",
]


GREETINGS = [
            f"Hello! I'm {BOT_NAME}, your HR assistant at {COMPANY_NAME}. How can I help you today?",
            f"Hi there! {BOT_NAME} here. Got an HR question? I'm happy to help.",
            f"Welcome to {BOT_NAME}, {COMPANY_NAME}'s HR assistant! What would you like to know?",
            f"Hey! I'm {BOT_NAME}. Ask me anything about HR at {COMPANY_NAME}.",
            f"Good to see you! I'm {BOT_NAME}. What HR query can I help you with today?",
]


GOODBYES = [
            f"Goodbye! If you have any HR-related doubts, {BOT_NAME} is always here to help.",
            f"Thanks for chatting with {BOT_NAME}! Reach out anytime you have an HR question.",
            "Take care! Come back anytime you need help with HR matters.",
            f"Glad I could help. Have a great day at {COMPANY_NAME}!",
            f"Bye for now! {BOT_NAME} will be here whenever you have an HR query.",
]


EXIT_WORDS = {
    "exit", "quit", "bye", "goodbye", "bye bye",
    "thanks", "thank you", "thankyou",
    "thanks bye", "thank you bye", "ok bye",
}


# ---------- Employee records, loaded from the JSON file ----------
with open(EMPLOYEES_PATH, encoding="utf-8") as f:
    EMPLOYEE_RECORDS = {str(e["EmpId"]): e for e in json.load(f)}
    
# converts it from a raw json to dict format.

