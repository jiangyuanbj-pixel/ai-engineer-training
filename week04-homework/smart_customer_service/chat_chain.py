from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.llms import Tongyi
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
import tools

from typing import List, Dict, Any
import asyncio

# 增加调试日志
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("llama_index").addHandler(logging.StreamHandler(stream=sys.stdout))


class ChatChain:
    def __init__(self):
        self.llm = None
        self.chain = None
        self.parser = StrOutputParser()
        self.tools_map = {
            "query_order": tools.query_order
        }

    async def initialize(self):
        """异步初始化"""
        # 初始化 LLM
        # self.llm = Tongyi(
        #     temperature=0.7,
        #     model_name="qwen-turbo"  # LangChain 0.3 推荐明确指定模型
        # )

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        llm = ChatOpenAI(
            base_url=base_url, api_key=api_key, model="deepseek-chat", temperature=0.7
        )

        # 绑定到 LLM
        self.llm = llm.bind_tools([tools.query_order])

        # 创建提示模板 - LangChain 0.3 风格
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个智能客服助手，请遵循以下规则：
                1. 友好、专业地回答用户问题
                2. 如果不确定答案，诚实地说不知道
                3. 保持回答简洁明了
                4. 根据对话历史提供连贯的回复
                5. 用中文回答""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )

        # 构建链 - LangChain 0.3 LCEL 语法
        # self.chain = (
        #     RunnablePassthrough.assign(
        #         history=RunnableLambda(self._format_history)
        #     )
        #     | self.prompt
        #     | self.llm
        #     | self.parser
        # )

    async def process_message(self, message: str, history: List[Dict] = None) -> str:
        """处理用户消息"""
        try:
            # 准备输入数据
            input_data = {"message": message, "raw_history": history or [], "tools_mesaage": []}

            # 异步调用链
            # response = await self.chain.ainvoke(input_data)
            # response = self.chain.invoke(input_data)

            max_rounds = 5
            for round_num in range(max_rounds):
                response = (
                    RunnablePassthrough.assign(
                        history=RunnableLambda(self._format_history)
                    )
                    | self.prompt
                    | self.llm
                ).invoke(input_data)

                if not response.tool_calls:  # 没有工具场景
                    return response.content

                #llm的回复要加进去
                input_data.get("tools_mesaage").append(response)

                # 有工具调用 → 执行每个工具，把结果返回给 LLM
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    print(f"[调用工具]: {tool_name}({tool_args})")
                    # 执行工具
                    tool_fn = self.tools_map[tool_name]
                    result = tool_fn.invoke(tool_args)

                    print(f"[工具返回]: {result}")
                    tm = ToolMessage(content=str(result), tool_call_id=tool_id)
                    input_data.get("tools_mesaage").append(tm)

            return "超过最大调用轮次，请重试。"

        except Exception as e:
            print(f"处理消息时出错: {e}")
            return "抱歉，我现在无法处理您的请求，请稍后再试。"

    def _format_history(self, input_data: Dict[str, Any]) -> List:
        """格式化历史消息为 LangChain 消息格式"""
        history = input_data.get("raw_history", [])

        tools = input_data.get("tools_mesaage", [])

        messages = []
        # 只保留最近5轮对话
        if len(history):
            recent_history = history[-5:] if len(history) > 5 else history
            for item in recent_history:
                messages.append(HumanMessage(content=item["user_message"]))
                messages.append(AIMessage(content=item["bot_reply"]))
        if len(tools) > 0:
            # 如果有tools消息， 先把当前用户问题加上去
            messages.append(HumanMessage(content=input_data.get('message')))
            for item in tools:
                messages.append(item)

        return messages

"""
BadRequestError('Error code: 400 - {\'error\': {\'message\': "Messages with role \'tool\' must be a response to a preceding message with \'tool_calls\'", \'type\': \'invalid_request_error\', \'param\': None, \'code\': \'invalid_request_error\'}}')
'订单id=1000000的详细信息如下：\n{\n  "order_id": "1000000",\n  "order_time": "2025-06-01 12:00:00",\n  "order_amount": 250.0,\n  "order_status": "\\u5df2\\u5b8c\\u6210",\n  "order_items": [\n    {\n      "item_id": "123456",\n      "item_name": "\\u5546\\u54c11",\n      "item_price": 50.0,\n      "item_quantity": 1\n    },\n    {\n      "item_id": "123457",\n      "item_name": "\\u5546\\u54c12",\n      "item_price": 100.0,\n      "item_quantity": 2\n    }\n  ],\n  "order_address": "\\u4e0a\\u6d77\\u5e02\\u5f90\\u6c47\\u533a\\u6f15\\u6cb3\\u6cfe\\u5f00\\u53d1\\u533a",\n  "order_remarks": "\\u65e0"\n}'
"""