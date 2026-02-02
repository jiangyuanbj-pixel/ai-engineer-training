from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    应用配置类
    通过环境变量加载配置，支持 .env 文件
    """
    PROJECT_NAME: str = "High-Concurrency Chat Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # 限流配置 (每秒请求数)
    RATE_LIMIT_PER_SECOND: int = 1000
    
    # 日志级别
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
