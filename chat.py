# =====================================================================
# chat.py  -  the terminal chat (the "front door")
# Run it with:  uv run chat.py
# It only does three things: read what you type, call ask(), print the reply.
# The thinking all happens in policy_agent.py.
# =====================================================================

import config
import random
from policy_agent import ask
from langchain_core.messages import AIMessage,HumanMessage


def main():
    
    print(f"\n{random.choice(config.GREETINGS)}\n")
    
    # history = the conversation so far (your messages + the agent's replies).
    # It is what lets the agent remember an employee ID you typed earlier,
    # and understand a short reply like "384" or "permanent".
    
    history = []
    
    while True:
        question = input("You: ").strip()
        
        # 'while True' -- repeats forever, until something says break.
        # 'input("You: ")' -- pauses and waits for you to type, then returns what you typed.
        # '.strip()' -- removes spaces at the start and end, so " hello " becomes "hello".
        
        
        
        cleaned = question.lower().strip("!.?, ")
        cleaned = " ".join(cleaned.split())

        if cleaned in config.EXIT_WORDS:   # leave the chat   # leave the chat
            print(f"{config.BOT_NAME}: {random.choice(config.GOODBYES)}\n")
            break
        # Clean the input before checking for exit words:
                #   .lower()       -> "Bye", "BYE" and "bYe" all become "bye"
                #   .strip("!.?,") -> removes punctuation at the ends, so "bye!" becomes "bye"
                #   split + join   -> collapses extra spaces, so "thank   you" becomes "thank you"
        
        
        if question == "":  # ignore empty Enter presses
            continue
        
        
        
        answer = ask(question,history= history)
        # It sends your question plus the history to the agent and waits for the reply text.
        
        
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content = answer))
        
        print(f"{config.BOT_NAME}: {answer}\n")
        
if __name__ == "__main__":
        main()
