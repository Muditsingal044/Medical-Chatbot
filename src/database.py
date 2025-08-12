from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class ChatLog(Base):
    __tablename__ = 'chat_logs'
    
    id = Column(Integer, primary_key=True)
    user_message = Column(String, nullable=False)
    bot_response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create database engine
engine = create_engine('sqlite:///chat_history.db')

# Create all tables
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_chat(user_message: str, bot_response: str):
    db = SessionLocal()
    try:
        chat_log = ChatLog(
            user_message=user_message,
            bot_response=bot_response
        )
        db.add(chat_log)
        db.commit()
        return chat_log
    finally:
        db.close()

def get_chat_history():
    db = SessionLocal()
    try:
        return db.query(ChatLog).all()
    finally:
        db.close()

def clear_chat_history():
    db = SessionLocal()
    try:
        db.query(ChatLog).delete()
        db.commit()
    finally:
        db.close()

def get_recent_context(limit=5):
    """Get recent chat history for context"""
    db = SessionLocal()
    try:
        return db.query(ChatLog).order_by(ChatLog.timestamp.desc()).limit(limit).all()
    finally:
        db.close()
