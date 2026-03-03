from sqlalchemy import Column, Integer, String, Index
from src.database import Base


class DatasetRow(Base):
    __tablename__ = "dataset_rows"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    label = Column(Integer, nullable=False)
    emotion = Column(String, index=True, nullable=False)

    __table_args__ = (
        Index("idx_emotion", "emotion"),
        # Index on text for full-text-search in SQLite might require FTS,
        # but a simple index works for 'LIKE' queries occasionally.
    )
