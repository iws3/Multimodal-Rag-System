# backend/app/db/base.py
from sqlalchemy.orm import declarative_base

# Create the declarative base that all models will use
Base = declarative_base()