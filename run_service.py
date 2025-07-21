import uvicorn
import asyncio
import sys
from core.settings import settings

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("api_service.api_service:app", host=settings.HOST, port=settings.PORT)