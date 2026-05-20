"""基于 LangGraph 的智能客服对话链

阶段一：基础对话系统 — Prompt → LLM → OutputParser
阶段二：多轮对话与工具调用 — LangGraph ReAct Agent
"""

import os
import logging
from typing import Dict, List, Any, Annotated
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import tools

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============ 状态定义 ============

class AgentState(TypedDict):
    """LangGraph Agent 状态"""
    messages: Annotated[list, add_messages]


class ChatChain:
    """智能客服对话链

    阶段一：基础 Prompt → LLM → OutputParser 链
    阶段二：LangGraph 多轮工具调用 Agent
    """

    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.chain = None  # 阶段一：基础链
        self.graph = None  # 阶段二：LangGraph
        self.parser = StrOutputParser()
        self.tools_map = {t.name: t for t in tools.all_tools}

    async def initialize(self):
        """初始化 LLM 和对话链"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.7,
        )

        # ============ 阶段一：基础对话链 ============
        self._build_basic_chain()

        # ============ 阶段二：LangGraph Agent ============
        self._build_agent_graph()

    # ----------------------------------------------------------------
    # 阶段一：基础对话系统（Prompt → LLM → OutputParser）
    # ----------------------------------------------------------------

    def _build_basic_chain(self):
        """构建基础对话链，注入当前时间让 LLM 理解相对时间"""
        now = datetime.now()
        weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]

        self.basic_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个智能客服助手，请遵循以下规则：
1. 友好、专业地回答用户问题
2. 如果不确定答案，诚实地说不知道
3. 保持回答简洁明了
4. 根据对话历史提供连贯的回复
5. 用中文回答

当前时间信息：{current_time}""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )

        self.chain = (
            self.basic_prompt
            | self.llm
            | self.parser
        )

    async def basic_chat(self, message: str, history: List[Dict] = None) -> str:
        """阶段一：基础对话（不支持工具调用）"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_history = self._format_history_to_messages(history or [])

        input_data = {
            "message": message,
            "history": formatted_history,
            "current_time": f"{now_str}，星期{['一','二','三','四','五','六','日'][datetime.now().weekday()]}",
        }

        response = await self.chain.ainvoke(input_data)
        return response

    # ----------------------------------------------------------------
    # 阶段二：LangGraph 多轮对话 + 工具调用
    # ----------------------------------------------------------------

    def _build_agent_graph(self):
        """构建 LangGraph ReAct Agent 图

        流程：START → agent → should_continue? → tools → agent (循环)
                                           → END
        """
        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(tools.all_tools)

        # 创建图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(tools.all_tools))

        # 设置入口
        workflow.set_entry_point("agent")

        # 条件边：agent 根据是否有 tool_calls 决定走向
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # tools 执行完后回到 agent
        workflow.add_edge("tools", "agent")

        # 编译图
        self.graph = workflow.compile()

    async def _agent_node(self, state: AgentState) -> dict:
        """Agent 节点：调用 LLM 决策"""
        messages = state["messages"]

        # 注入系统提示（含当前时间）
        now = datetime.now()
        weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
        system_msg = SystemMessage(
            content=f"""你是一个智能客服助手，请遵循以下规则：
1. 友好、专业地回答用户问题
2. 如果不确定答案，诚实地说不知道
3. 保持回答简洁明了
4. 根据对话历史提供连贯的回复
5. 用中文回答
6. 当用户提到"查订单"但没提供订单号时，请追问订单号
7. 当用户提到相对时间（如"昨天"、"上周"），请调用 get_current_time 获取当前时间来推断

当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，星期{weekday}"""
        )

        # 确保系统提示在消息列表开头
        all_messages = [system_msg] + messages

        response = await self.llm_with_tools.ainvoke(all_messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续调用工具"""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    async def agent_chat(self, message: str, history: List[Dict] = None) -> str:
        """阶段二：LangGraph Agent 对话（支持多轮工具调用）"""
        # 构建消息列表
        messages = []

        # 加入历史对话
        if history:
            for item in history[-5:]:
                messages.append(HumanMessage(content=item["user_message"]))
                bot_reply = item.get("bot_reply", "")
                if bot_reply:
                    messages.append(AIMessage(content=bot_reply))

        # 加入当前用户消息
        messages.append(HumanMessage(content=message))

        # 调用 LangGraph
        result = await self.graph.ainvoke({"messages": messages})

        # 提取最后的 AI 回复
        last_message = result["messages"][-1]
        return last_message.content

    # ----------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------

    async def process_message(self, message: str, history: List[Dict] = None) -> str:
        """处理用户消息（使用阶段二 Agent）"""
        try:
            return await self.agent_chat(message, history)
        except Exception as e:
            logger.error(f"处理消息时出错: {e}", exc_info=True)
            return "抱歉，我现在无法处理您的请求，请稍后再试。"

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    def _format_history_to_messages(self, history: List[Dict]) -> list:
        """将历史记录格式化为 LangChain 消息列表"""
        messages = []
        recent = history[-5:] if len(history) > 5 else history
        for item in recent:
            messages.append(HumanMessage(content=item["user_message"]))
            bot_reply = item.get("bot_reply", "")
            if bot_reply:
                messages.append(AIMessage(content=bot_reply))
        return messages
