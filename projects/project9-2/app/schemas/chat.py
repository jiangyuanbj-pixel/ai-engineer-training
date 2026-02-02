from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """
    聊天请求模型
    """
    message: str
    client_id: str

class ChatResponse(BaseModel):
    """
    聊天响应模型
    """
    status: str
    message: str
    client_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
