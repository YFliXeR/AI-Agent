from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations"""
    print("Initializing The calculator Tool...")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def sayhello(name: str) -> str:
    """Useful for greeting a user"""
    print("Initializing The sayhello Tool...")
    return f"Hello {name}, nice meeting you"

def main():
    model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai") #Choosing the model

    tools = [calculator, sayhello]
    agent_executor = create_react_agent(model, tools) #Initializing the prebuilt agent and giving it the model and tools we want

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip() #Using strip to remove any unnecessary white space at the start or end

        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="") #end is there to remove the new line that python makes by default so the assistant's answer wouldnt be in a new line 
    
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}): #Passing the messages to our agent
            if "agent" in chunk and "messages" in chunk["agent"]: #Is the current chunk a response from the agent? and if there's any message in that particular response
                for message in chunk["agent"]["messages"]: #Looping through the messages and printing its content
                    print(message.content, end="")
    print()

if __name__ == "__main__": #If this python file is executed directly then only run the main() function
    main()