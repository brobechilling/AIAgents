from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from core.settings import settings
from memory import initialize_database
from memory.postgres import get_postgres_saver
import logging

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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GOOGLE_API_KEY)
    memory = await create_database_checkpointer(conn)

    class BasicChatState(TypedDict): 
        messages: Annotated[list, add_messages]

    async def chatbot(state: BasicChatState): 
        return {
            "messages": [await llm.ainvoke(state["messages"])]
        }

    graph = StateGraph(BasicChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge("chatbot", END)
    graph.set_entry_point("chatbot")

    app = graph.compile(checkpointer=memory)

    return app


