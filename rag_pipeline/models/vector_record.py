"""
VectorRecord entity model
Represents an entry in Qdrant Cloud containing the embedding vector and associated metadata
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class VectorRecord(BaseModel):
    """
    Represents an entry in Qdrant Cloud containing the embedding vector and associated metadata
    for retrieval
    """
    chunk_id: str
    content: str
    source_url: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

    def __str__(self):
        return f"VectorRecord(chunk_id={self.chunk_id}, source_url={self.source_url}, chapter={self.chapter})"

    def __repr__(self):
        return self.__str__()