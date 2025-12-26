"""
ContentChunk entity model
Represents a segment of book content that has been chunked according to the strategy
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ContentChunk(BaseModel):
    """
    Represents a segment of book content that has been chunked according to the strategy,
    with metadata about its source location
    """
    id: str
    book_content_id: str
    content: str
    chunk_index: int
    semantic_boundary: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    source_url: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

    def __str__(self):
        return f"ContentChunk(id={self.id}, book_content_id={self.book_content_id}, chunk_index={self.chunk_index})"

    def __repr__(self):
        return self.__str__()