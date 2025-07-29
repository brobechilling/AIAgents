from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage, ToolMessage
from core.settings import settings
from memory.postgres import get_postgres_saver
import logging
from tool_nodes.rag_tools import retriever_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_database_checkpointer(conn):
    """Create and return a PostgreSQL checkpointer using a provided connection."""
    try:
        # Pass the connection to saver logic
        saver = get_postgres_saver(conn)
        if hasattr(saver, "setup"):
            await saver.setup()
        return saver
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise

async def create_chatbot(conn):
    # Tool setup
    tools = [retriever_tool]
    tools_dict = {tool.name: tool for tool in tools}

    llm = ChatGoogleGenerativeAI( 
        model="gemini-2.0-flash", 
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0).bind_tools(tools)

    system_prompt = """
You are an intelligent AI assistant helping users understand the VroomVroom User Manual.

1. Always use the retriever tool to search for accurate answers.
2. Interpret and **summarize** retrieved content in your own words unless directly quoting.
3. Be clear and detailed, even for short questions.
4. If the answer is unclear or needs context, ask a follow-up or explain what's missing.

Avoid just saying "I don't know" unless no document truly matches.
"""

    memory = await create_database_checkpointer(conn)

    class ChatState(TypedDict): 
        messages: Annotated[list[BaseMessage], add_messages]

    # LLM Agent
    async def llm_node(state: ChatState) -> ChatState:
        """Function to call the LLM with the current state."""
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        # Fix empty content error
        for m in messages:
            if not getattr(m, "content", None):
                m.content = "No content."
        message = llm.invoke(messages)
        return {'messages': [message]}

    # Retriever Agent
    async def tool_node(state: ChatState) -> ChatState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
            
            if not t['name'] in tools_dict: # Checks if a valid tool is present
                print(f"\nTool: {t['name']} does not exist.")
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                print(f"Result length: {len(str(result))}")
                


            # Appends the Tool Message
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        print("Tools Execution Complete. Back to the model!")
        return {'messages': results}
        

    def should_continue(state: ChatState) -> bool:
        """Check if the last message contains tool calls."""
        last_msg = state["messages"][-1]
        return hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0

    # Graph setup
    graph = StateGraph(ChatState)
    graph.add_node("llm", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {True: "tool_node", False: END})
    graph.add_edge("tool_node", "llm")

    app = graph.compile(checkpointer=memory)

    return app