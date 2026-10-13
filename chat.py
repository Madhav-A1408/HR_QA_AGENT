# =====================================================================
# chat.py  -  the terminal chat (the "front door")
# Run it with:  uv run chat.py
# It only does three things: read what you type, call ask(), print the reply.
# The thinking all happens in policy_agent.py.
# =====================================================================

from policy_agent import ask
from langchain_core.messages import AIMessage,HumanMessage


def main():
    print("\nHi! Ask me a policy/HR question (type 'exit' to quit).\n")
    
    # history = the conversation so far (your messages + the agent's replies).
    # It is what lets the agent remember an employee ID you typed earlier,
    # and understand a short reply like "384" or "confirmed".
    
    history = []
    
    while True:
        question = input("You: ").strip()
        
        # 'while True' -- repeats forever, until something says break.
        # 'input("You: ")' -- pauses and waits for you to type, then returns what you typed.
        # '.strip()' -- removes spaces at the start and end, so " hello " becomes "hello".
        
        
        if question.lower() in {"quit", "exit"}:
            break
        #.lower() makes it case-insensitive, so Exit, EXIT and quit all work. 
        # break leaves the loop, so the program ends.
        
        
        if question == "":  # ignore empty Enter presses
            continue
        
        
        
        answer = ask(question,history= history)
        # It sends your question plus the history to the agent and waits for the reply text.
        
        
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content = answer))
        print(f"Agent: {answer}\n")
        
if __name__ == "__main__":
        main()
