"""
BookContent entity model
Represents the original book content extracted from the website
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class BookContent(BaseModel):
    """
    Represents the original book content extracted from the website
    """
    id: str
    url: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

    def __str__(self):
        return f"BookContent(id={self.id}, title={self.title}, url={self.url})"

    def __repr__(self):
        return self.__str__()