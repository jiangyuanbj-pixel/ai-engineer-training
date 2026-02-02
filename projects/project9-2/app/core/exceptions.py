class ServiceError(Exception):
    """服务基础异常"""
    pass

class RateLimitExceeded(ServiceError):
    """限流异常"""
    pass

class CacheError(ServiceError):
    """缓存操作异常"""
    pass

class LLMError(ServiceError):
    """LLM 调用异常"""
    pass
