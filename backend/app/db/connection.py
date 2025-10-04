# backend/app/db/connection.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config.setting import DATABASE_URL
import asyncio # For running async functions in a sync context (for init_db)
import sqlalchemy
from sqlalchemy import text  # Import text for raw SQL queries
from app.db import models  # Ensure all models are registered with Base.metadata

# --- 1. The Database Engine ---
# create_async_engine: This is our connection manager to the database.
# DATABASE_URL: The connection string from our settings.
# echo=False: If True, SQLAlchemy will print all SQL it executes to the console.
#             Useful for debugging, but verbose for production.
engine = create_async_engine(DATABASE_URL, echo=True)  # Enable SQL logging

# Import Base and models
from app.db.base import Base
from app.db.models import Document, Chunk, Query  # Import all models to ensure they are registered

# --- 3. Async Session Local Factory ---
# sessionmaker: This is a factory that will produce new AsyncSession objects.
# Each session represents a conversation with the database.
# autocommit=False, autoflush=False: Gives us explicit control over when to commit changes.
# bind=engine: Connects this session factory to our database engine.
# class_=AsyncSession: Specifies that the sessions produced will be asynchronous.
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# --- 4. FastAPI Dependency for Database Session ---
# This is a special function designed for FastAPI's dependency injection system.
# When an API endpoint needs a database session, it will ask for 'db: AsyncSession = Depends(get_db)'.
# FastAPI will then call this function, manage the session's lifecycle (creation, yielding, closing).
async def get_db():
    db = AsyncSessionLocal() # Create a new asynchronous database session
    try:
        yield db # 'yield' means this function is a context manager.
                 # The session 'db' is provided to the calling function.
    finally:
        await db.close() # Ensure the session is closed after the request is processed,
                         # releasing the database connection.

# --- 5. Database Initialization Function ---
# This function will create our database tables and ensure the 'vector' extension is enabled.
async def init_db():
    async with engine.begin() as conn: # Get an asynchronous connection and start a transaction.
                                      # 'begin()' ensures all operations are atomic.
        print("\n--- Initializing Database Schema ---")

        # Debug: Print the DATABASE_URL to confirm it's loaded correctly
        print(f"Using DATABASE_URL: {DATABASE_URL}")

        # Test the database connection
        print("Testing database connection...")
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1;"))
            print("Database connection successful.")
        except Exception as e:
            print(f"ERROR: Could not connect to the database. {e}")
            return

        # Ensure connection remains open for 'vector' extension creation
        print("1. Ensuring 'vector' extension exists...")
        try:
            async with engine.begin() as vector_conn:
                await vector_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print("   'vector' extension enabled successfully (or already existed).")
        except Exception as e:
            print(f"   WARNING: Could not create 'vector' extension. Error: {e}")
            print("   Skipping 'vector' extension creation.")

        # Log table creation
        print("2. Creating tables based on models...")
        try:
            async with engine.begin() as table_conn:
                await table_conn.run_sync(Base.metadata.create_all)
                print("   Database tables created/updated successfully.")
        except Exception as e:
            print(f"   ERROR: Failed to create tables. Error: {e}")
            print("   Please check your model definitions and database connection.")

        # Ensure connection is open for table verification
        print("3. Verifying table creation...")
        try:
            async with engine.begin() as verify_conn:
                result = await verify_conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
                tables = [row[0] for row in result]
                print(f"   Tables in the database: {tables}")
        except Exception as e:
            print(f"   WARNING: Could not verify tables. Error: {e}")
        
        print("--- Database Schema Initialization Complete ---\n")

# --- 6. Direct Script Execution (for easy setup) ---
# This block allows you to run 'python app/db/connection.py' directly from your terminal
# to set up your database without starting the full FastAPI application.
if __name__ == "__main__":
    print("Attempting to initialize database via direct script execution...")
    asyncio.run(init_db()) # Run the async init_db function
    print("Database initialization script finished.")