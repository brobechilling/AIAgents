import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Configuration settings for the application."""
    
    # LLM Configuration
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    # PostgreSQL Configuration
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD") 
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST") 
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT") 
    POSTGRES_DB = os.environ.get("POSTGRES_DB") 
    POSTGRES_SSLMODE = os.environ.get("POSTGRES_SSLMODE") 

    # API Service
    HOST = os.environ.get("API_SERVICE_HOST")
    PORT = os.environ.get("API_SERVICE_PORT")

settings = Settings()