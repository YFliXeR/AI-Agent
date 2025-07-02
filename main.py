from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def available_tools():
    """"Useful for whenever user types only 1 in chat to show all the available tools, ignore those 2 tools (available_tools and sayhello) in your answer"""

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations (return only the operation that users asks)"""
    print("Initializing The calculator Tool...\n")
    result = (
        f"The sum of {a} and {b} is {a + b}"
        f"The difference between {a} and {b} is {a - b}"
        f"The product of {a} and {b} is {a * b}"
        f"The divison of {a} and {b} is {a / b if b != 0 else 'undefined'}"
    )
    return result 

@tool
def sayhello(name: str) -> str:
    """Useful for greeting a user"""
    print("Initializing The Specialized Greeting Tool...\n")
    return f"Hello {name}, nice meeting you"


@tool
def web_search(search_query: str) -> str:
    """"Useful for free online search about anything. Never change or rephrase the return value, it's ready as an answer to the user."""
    print("Initializing DuckDuckGo Search Tool...\n")
    results = DuckDuckGoSearchResults(output_format="list").invoke(search_query)
    return f"""Your Search Results Are Ready!📑:
            Top result🏆: 
            Title = {results[0]["title"]}
            Link = {results[0]["link"]}
            Context = {results[0]["snippet"]}"""


def main():
    model = init_chat_model(model="gemini-2.5-pro", model_provider="google_genai") #Choosing the model

    tools = [calculator, sayhello, web_search, available_tools]
    agent_executor = create_react_agent(model, tools) #Initializing the prebuilt agent and giving it the model and tools we want

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print(f"Type 1 to show the available tools.")

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