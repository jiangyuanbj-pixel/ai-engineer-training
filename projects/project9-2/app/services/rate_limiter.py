from fastapi import HTTPException, status
from app.services.cache_service import cache_service
from app.core.config import settings
from app.core.exceptions import RateLimitExceeded

class RateLimiter:
    """
    简单的 Redis 固定窗口限流器
    """
    @staticmethod
    async def check_rate_limit(key: str, limit: int = settings.RATE_LIMIT_PER_SECOND, window: int = 1):
        """
        检查是否超过限流阈值
        :param key: 限流键 (通常是 IP 或 UserID)
        :param limit: 时间窗口内的最大请求数
        :param window: 时间窗口大小 (秒)
        """
        if not cache_service.redis:
            return # 如果 Redis 未连接，默认通过 (或抛出错误，视策略而定)

        # 构建 Redis Key
        redis_key = f"rate_limit:{key}"
        
        # 使用 Redis 管道保证原子性
        async with cache_service.redis.pipeline() as pipe:
            try:
                await pipe.incr(redis_key)
                await pipe.expire(redis_key, window)
                result = await pipe.execute()
                
                request_count = result[0]
                
                if request_count > limit:
                    raise RateLimitExceeded(f"Rate limit exceeded: {limit} requests per {window} second(s).")
            except RateLimitExceeded:
                raise
            except Exception as e:
                # 降级策略：如果限流出错，允许通过，但记录日志
                from app.core.logger import logger
                logger.error("Rate limiter error", error=str(e))
                pass

rate_limiter = RateLimiter()
