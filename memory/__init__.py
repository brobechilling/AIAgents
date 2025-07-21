from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from memory.postgres import get_postgres_saver

def initialize_database(conn) -> AsyncPostgresSaver:
    """
    Initialize the POSTGRES database checkpointer using a provided connection.
    Returns an initialized AsyncCheckpointer instance.
    """
    return get_postgres_saver(conn)

__all__ = ["initialize_database"]