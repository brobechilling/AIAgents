from core.settings import settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

def validate_postgres_config() -> None:
    """
    Validate that all required PostgreSQL configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )
    
def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
        f"?sslmode={settings.POSTGRES_SSLMODE}"
    )

def get_postgres_saver(conn) -> AsyncPostgresSaver:
    """Initialize and return a PostgreSQL saver instance using a provided connection."""
    validate_postgres_config()
    return AsyncPostgresSaver(conn)

