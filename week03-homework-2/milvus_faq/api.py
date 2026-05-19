"""FAQ 检索系统 — FastAPI 接口 + 热更新"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .main import FAQEngine, FAQ_DIR, WATCH_INTERVAL

# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    response: str

# ============ 全局引擎 + 热更新线程 ============

engine = FAQEngine()
_stop = threading.Event()


def _watch():
    """后台线程：定期检测 FAQ 文件变更并自动重建索引"""
    while not _stop.is_set():
        _stop.wait(WATCH_INTERVAL)
        if _stop.is_set():
            break
        try:
            engine.check_rebuild()
        except Exception as e:
            print(f"❌ 热更新失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时构建索引 + 启动热更新线程"""
    try:
        engine.build_index()
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")

    _stop.clear()
    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    print(f"📡 热更新守护线程已启动（间隔 {WATCH_INTERVAL}s）")

    yield

    _stop.set()


# ============ API 路由 ============

app = FastAPI(title="FAQ 检索系统", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """多轮对话检索 FAQ"""
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")
    try:
        response = engine.chat(req.message)
        return ChatResponse(message=req.message, response=response)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/chat/reset")
def reset_chat():
    """重置对话历史"""
    engine.reset_chat()
    return {"message": "对话历史已重置"}


@app.post("/index/rebuild")
def rebuild():
    """手动触发索引重建"""
    try:
        engine.build_index()
        return {"message": "索引重建成功"}
    except Exception as e:
        raise HTTPException(500, f"重建失败: {e}")
