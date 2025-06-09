from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import speech_recognition as sr
import pyttsx3

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str

@tool
def update(state: AgentState, content: str) -> AgentState:
    """Updates the document with the provided content."""
    state["document_content"] = content
    return state

@tool
def save(state: AgentState, filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    content = state.get("document_content", "")
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]

model = ChatOpenAI(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
).bind_tools(tools)

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening for your input... (speak now)")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"\n👤 USER (voice): {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{state.get('document_content', '')}
    """)

    voice_mode = state.get("voice_mode", False)

    # Always use the selected input mode for user input, including the first user turn
    if not state["messages"]:
        # First user turn: use selected input mode
        if voice_mode:
            user_input = get_voice_input()
            if not user_input:
                user_input = "Please try speaking again."
        else:
            user_input = input("\nWhat would you like to create? ")
            print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    else:
        # All subsequent turns: use the selected input mode
        if voice_mode:
            user_input = get_voice_input()
            if not user_input:
                user_input = "Please try speaking again."
        else:
            user_input = input("\nWhat would you like to do with the document? ")
            print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    # Debug: Print the raw response object for troubleshooting
    print(f"\n[DEBUG] Raw AI response: {response}")

    # Fix: If response.content is empty, print a warning and skip speaking
    if not response.content or not response.content.strip():
        print("\n🤖 AI: [No response from AI. Please try again or check your prompt/API key.]")
    else:
        print(f"\n🤖 AI: {response.content}")
        if voice_mode:
            try:
                speak_text(response.content)
            except Exception as e:
                print(f"[Voice output error: {e}]")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response], "document_content": state.get("document_content", ""), "voice_mode": voice_mode}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    mode = ""
    while mode not in ["1", "2"]:
        print("Choose input mode:")
        print("1. Text (keyboard)")
        print("2. Voice (microphone)")
        mode = input("Enter 1 or 2: ").strip()
    voice_mode = mode == "2"
    state = {"messages": [], "document_content": "", "voice_mode": voice_mode}
    if voice_mode:
        # On first turn, force voice input and print AI response
        state = our_agent(state)
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()

