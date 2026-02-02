from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.connection_manager import connection_manager
from app.services.chat_service import chat_service
from app.services.rate_limiter import rate_limiter
from app.core.logger import logger

router = APIRouter()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket 连接端点
    """
    await connection_manager.connect(client_id, websocket)
    try:
        while True:
            # 保持连接活跃，监听消息 (虽然本场景主要是单向推送)
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), client_id=client_id)
        connection_manager.disconnect(client_id)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    聊天 HTTP 端点
    接收请求 -> 检查限流 -> 触发后台任务 -> 返回确认
    """
    # 1. 检查 WebSocket 连接状态
    if request.client_id not in connection_manager.active_connections:
        raise HTTPException(status_code=400, detail="Client is not connected to WebSocket. Please connect to /ws/{client_id} first.")

    # 2. 限流检查
    await rate_limiter.check_rate_limit(request.client_id)

    # 3. 后台执行任务
    background_tasks.add_task(chat_service.process_chat_request, request.message, request.client_id)

    logger.info("Chat request accepted", client_id=request.client_id)

    return ChatResponse(
        status="accepted",
        message="Request accepted, processing in background.",
        client_id=request.client_id
    )
