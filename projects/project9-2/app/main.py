import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.core.config import settings
from app.core.logger import configure_logging, logger
from app.core.exceptions import ServiceError
from app.api.endpoints import router as api_router
from app.services.cache_service import cache_service

# 1. 配置日志
configure_logging()

# 2. 生命周期管理 (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时连接 Redis，关闭时断开连接
    """
    logger.info("Starting up application...")
    await cache_service.connect()
    yield
    logger.info("Shutting down application...")
    await cache_service.close()

# 3. 初始化 FastAPI 应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 4. 中间件：Request ID & 日志绑定
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

# 5. 中间件：CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. 注册路由
app.include_router(api_router)

# 7. 全局异常处理
@app.exception_handler(ServiceError)
async def service_exception_handler(request: Request, exc: ServiceError):
    logger.error("Service error occurred", error=str(exc))
    return Response(content=str(exc), status_code=500)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
