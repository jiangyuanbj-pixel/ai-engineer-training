from typing import Dict
from fastapi import WebSocket

class ConnectionManager:
    """
    WebSocket 连接管理器
    管理活跃连接，支持单播和广播
    """
    def __init__(self):
        # 存储活跃连接: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        """接受并注册新的 WebSocket 连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        """移除连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: dict):
        """向指定客户端发送消息"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        """广播消息给所有客户端"""
        for connection in self.active_connections.values():
            await connection.send_json(message)

# 全局实例
connection_manager = ConnectionManager()
