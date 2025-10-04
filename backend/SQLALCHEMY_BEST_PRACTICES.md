# Common SQLAlchemy Issues and Best Practices

## The Problem We Encountered

We faced an issue where SQLAlchemy wasn't creating tables in the database despite successful connection. This was caused by two main issues:

1. **Circular Import Problem**
   - `models.py` was importing `Base` from `connection.py`
   - `connection.py` was importing models from `models.py`
   - This circular dependency prevented proper model registration with SQLAlchemy

2. **Model Registration Issue**
   - Due to the circular import, the models weren't properly registered with the `Base` class
   - This caused SQLAlchemy to think there were no tables to create

## Best Practices to Avoid These Issues

### 1. Proper Project Structure
```
app/
├── db/
│   ├── base.py          # Contains ONLY the Base declaration
│   ├── connection.py    # Database connection and session management
│   └── models.py        # All your SQLAlchemy models
```

### 2. Correct Import Order
```python
# 1. base.py - Define your Base
from sqlalchemy.orm import declarative_base
Base = declarative_base()

# 2. models.py - Import Base from base.py
from app.db.base import Base
# Define your models...

# 3. connection.py - Import both Base and models
from app.db.base import Base
from app.db.models import Document, Chunk, Query  # Import all models
```

### 3. Model Registration Checklist
- ✓ Place `Base = declarative_base()` in a separate `base.py` file
- ✓ Import models explicitly in your database initialization code
- ✓ Ensure all models inherit from the same `Base` instance
- ✓ Import all models before calling `Base.metadata.create_all()`

### 4. Database Connection Best Practices
```python
# In your connection.py
async def init_db():
    # 1. Test connection first
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 2. Create extensions if needed
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:
        print(f"Extension creation failed: {e}")

    # 3. Create tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"Table creation failed: {e}")

    # 4. Verify tables were created
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables"
            ))
            tables = [row[0] for row in result]
            print(f"Created tables: {tables}")
    except Exception as e:
        print(f"Verification failed: {e}")
```

### 5. Environment Variables
- Always use environment variables for database credentials
- Create an `.env.example` file with placeholder values
- Document required environment variables
- Validate environment variables at startup

### 6. Debugging Tips
- Enable SQLAlchemy echo mode to see SQL queries:
  ```python
  engine = create_async_engine(DATABASE_URL, echo=True)
  ```
- Check model registration:
  ```python
  print([cls.__name__ for cls in Base.__subclasses__()])
  ```
- Verify database URL before connection:
  ```python
  print(f"Using DATABASE_URL: {DATABASE_URL}")
  ```

## Common Gotchas

1. **Circular Imports**
   - Keep `Base` declaration separate from models and connection code
   - Use a dedicated `base.py` file

2. **Missing Model Imports**
   - Always import all models where you call `create_all()`
   - Simply defining models isn't enough; they must be imported

3. **Connection String Format**
   - For asyncpg, use: `postgresql+asyncpg://`
   - Include SSL requirements: `?ssl=require` for cloud databases

4. **Session Management**
   - Use async context managers with `async with`
   - Close connections properly in `finally` blocks

## Additional Resources

- SQLAlchemy Documentation: https://docs.sqlalchemy.org/
- FastAPI with SQLAlchemy: https://fastapi.tiangolo.com/tutorial/sql-databases/
- AsyncPG Documentation: https://magicstack.github.io/asyncpg/current/