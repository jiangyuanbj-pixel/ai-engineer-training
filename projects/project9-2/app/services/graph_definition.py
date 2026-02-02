import asyncio
import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import StateGraph, END

# 1. 定义状态 (State)
class AgentState(TypedDict):
    # messages 列表存储对话历史
    # operator.add 表示当新消息到达时，将其追加到现有列表中
    messages: Annotated[List[BaseMessage], operator.add]

# 2. 定义节点 (Node)
async def call_model(state: AgentState):
    """
    模拟调用大语言模型 (LLM) 的节点。
    """
    # 模拟网络延迟
    await asyncio.sleep(0.5)
    
    responses = [
        "我已收到您的请求。",
        "正在分析数据...",
        "这是基于您输入生成的最终结果 (Processed)."
    ]
    model = FakeListChatModel(responses=responses)
    
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

async def process_data(state: AgentState):
    """
    模拟数据处理节点。
    """
    await asyncio.sleep(0.5)
    return {"messages": [AIMessage(content="数据处理完成。")]}

# 3. 构建图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("processor", process_data)

# 设置入口点
workflow.set_entry_point("agent")

# 添加边
workflow.add_edge("agent", "processor")
workflow.add_edge("processor", END)

# 编译图
graph = workflow.compile()
