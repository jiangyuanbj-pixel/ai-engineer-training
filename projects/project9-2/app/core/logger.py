import logging
import sys
import structlog
from app.core.config import settings

def configure_logging():
    """
    配置结构化日志 (structlog)
    """
    
    # 设置标准库日志的基本配置
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars, # 合并上下文变量 (如 request_id)
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if not settings.DEBUG else structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()
