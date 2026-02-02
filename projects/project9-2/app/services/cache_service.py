import redis.asyncio as redis
from typing import Optional
from app.core.config import settings
from app.core.logger import logger

class CacheService:
    """
    Redis 缓存服务
    封装 Redis 操作，提供异步接口
    """
    def __init__(self):
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """连接到 Redis"""
        try:
            self.redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    async def close(self):
        """关闭连接"""
        if self.redis:
            await self.redis.close()
            logger.info("Closed Redis connection")

    async def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        if not self.redis:
            return None
        return await self.redis.get(key)

    async def set(self, key: str, value: str, expire: int = 3600):
        """设置缓存"""
        if self.redis:
            await self.redis.set(key, value, ex=expire)

# 全局单例
cache_service = CacheService()
