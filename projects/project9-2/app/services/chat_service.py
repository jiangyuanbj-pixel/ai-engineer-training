import hashlib
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.messages import HumanMessage
from app.services.graph_definition import graph
from app.services.cache_service import cache_service
from app.services.connection_manager import connection_manager
from app.core.logger import logger
from app.core.exceptions import LLMError

class ChatService:
    """
    聊天业务服务
    处理核心逻辑：缓存检查、图执行、重试机制、消息推送
    """

    @staticmethod
    def _generate_cache_key(message: str) -> str:
        """生成缓存 Key (基于消息内容的 Hash)"""
        hash_obj = hashlib.md5(message.encode())
        return f"chat_cache:{hash_obj.hexdigest()}"

    @retry(
        stop=stop_after_attempt(3), # 重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10), # 指数退避
        retry=retry_if_exception_type(Exception), # 捕获所有异常进行重试 (生产环境应更具体)
        reraise=True
    )
    async def _execute_graph_with_retry(self, message: str, client_id: str):
        """
        执行 LangGraph 工作流 (带重试机制)
        """
        inputs = {"messages": [HumanMessage(content=message)]}
        accumulated_content = []

        logger.info("Starting workflow execution", client_id=client_id)
        
        try:
            # 实时流式传输事件
            async for event in graph.astream_events(inputs, version="v2"):
                kind = event["event"]
                
                # 节点开始
                if kind == "on_chain_start":
                    name = event.get("name")
                    if name and name in ["agent", "processor"]:
                        await connection_manager.send_message(client_id, {
                            "type": "status",
                            "content": f"节点 '{name}' 已启动。"
                        })
                
                # 节点完成 (获取输出)
                elif kind == "on_chain_end":
                    name = event.get("name")
                    if name and name in ["agent", "processor"]:
                        data = event.get("data", {}).get("output")
                        if data and isinstance(data, dict) and "messages" in data:
                            last_msg = data["messages"][-1]
                            content = last_msg.content
                            accumulated_content.append(content)
                            
                            await connection_manager.send_message(client_id, {
                                "type": "message",
                                "content": content
                            })
                
                # LLM 流式输出 (如果有)
                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                         await connection_manager.send_message(client_id, {
                            "type": "token",
                            "content": content
                        })

            return "\n".join(accumulated_content)

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e), client_id=client_id)
            raise LLMError(f"Workflow execution failed: {str(e)}")

    async def process_chat_request(self, message: str, client_id: str):
        """
        处理聊天请求的主入口
        1. 检查缓存
        2. 执行工作流 (如果未命中缓存)
        3. 保存缓存
        """
        cache_key = self._generate_cache_key(message)
        
        # 1. 检查缓存
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            logger.info("Cache hit", client_id=client_id)
            await connection_manager.send_message(client_id, {
                "type": "status",
                "content": "结果已从缓存加载。"
            })
            await connection_manager.send_message(client_id, {
                "type": "message",
                "content": cached_result
            })
            await connection_manager.send_message(client_id, {
                "type": "status",
                "content": "工作流已完成 (Cached)。"
            })
            return

        # 2. 缓存未命中，执行工作流
        try:
            logger.info("Cache miss, executing workflow", client_id=client_id)
            await connection_manager.send_message(client_id, {
                "type": "status",
                "content": "工作流已启动。"
            })
            
            final_result = await self._execute_graph_with_retry(message, client_id)
            
            # 3. 保存结果到缓存
            if final_result:
                await cache_service.set(cache_key, final_result)
            
            await connection_manager.send_message(client_id, {
                "type": "status",
                "content": "工作流已完成。"
            })
            
        except Exception as e:
            await connection_manager.send_message(client_id, {
                "type": "error",
                "content": f"处理请求时发生错误: {str(e)}"
            })

chat_service = ChatService()
