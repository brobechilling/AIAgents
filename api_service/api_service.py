from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from agents import chatbot
from memory.postgres import get_postgres_connection_string

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncConnectionPool(conninfo=get_postgres_connection_string(), kwargs={"autocommit": True}) as pool:
        app.state.pg_pool = pool
        yield
    # (Optional) Add cleanup code here

app = FastAPI(lifespan=lifespan)

class Chat(BaseModel):
    message: str = None
    thread_id: int = 1

@app.post("/chat")
async def chat(chat: Chat):
    if not chat.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    config = {"configurable": {"thread_id": chat.thread_id}}
    async with app.state.pg_pool.connection() as conn:
        agent = await chatbot.create_chatbot(conn)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=chat.message)]
        }, config=config)

    if result and "messages" in result:
        return {"response": result["messages"][-1].content}
    else:
        raise HTTPException(status_code=500, detail="Error processing the request")

