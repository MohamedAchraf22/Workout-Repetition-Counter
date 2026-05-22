from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class WorkoutResult(Base):
    __tablename__ = "workout_results"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    video_path = Column(String)
    count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)