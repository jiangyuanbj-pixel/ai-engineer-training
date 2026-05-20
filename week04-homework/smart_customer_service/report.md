# 第四周作业 — 智能客服系统

## 一、系统概述

基于 LangChain 1.0+ 与 LangGraph 构建的多轮对话智能客服，支持工具调用与相对时间推断。

**技术栈**：FastAPI + LangChain 1.3 + LangGraph 1.2 + DeepSeek LLM

## 二、阶段一：基础对话系统

使用 LCEL 构建 Prompt → LLM → OutputParser 链，支持相对时间理解。

**核心实现**：在 system prompt 中注入当前时间，使 LLM 能推断"昨天""上周"等相对时间。

```python
self.chain = (
    self.basic_prompt    # 含 {current_time} 占位符
    | self.llm
    | self.parser
)
```

**示例**：
- 用户："我昨天下的单，什么时候能到？" → LLM 根据当前时间推断出昨天的具体日期并回复

## 三、阶段二：多轮对话与工具调用

使用 LangGraph 构建 ReAct Agent，支持自动工具调用与多轮追问。

### 流程图

```
START → agent(LLM决策) → 有 tool_calls?
                            ├── 是 → tools(执行) → agent(循环)
                            └── 否 → END
```

### 工具列表

| 工具 | 功能 | 说明 |
|------|------|------|
| `get_current_time` | 获取当前时间 | 支持相对时间推断（"昨天""上周"） |
| `query_order` | 查询订单详情 | 传入订单号，返回订单信息 |
| `apply_refund` | 申请退款 | 含重复退款校验、状态校验 |

### 多轮追问机制

- 用户："查订单" → LLM 判断缺少订单号 → 回复"请提供订单号"
- 用户："订单号是1000000" → LLM 调用 `query_order` → 返回订单信息

由 LLM 自身推理能力驱动，无需硬编码状态机。

## 四、文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | FastAPI 入口，定义 `/chat`、`/health`、`/session` 等接口 |
| `chat_chain.py` | 核心对话逻辑：阶段一 LCEL 链 + 阶段二 LangGraph Agent |
| `tools.py` | 工具定义：`get_current_time`、`query_order`、`apply_refund` |
| `session_manager.py` | 会话管理：历史记录、会话清理、统计 |
| `test_client.py` | 测试客户端，覆盖阶段一/二场景 |
| `report.md` | 本报告 |

## 五、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 对话接口，参数：`session_id`、`message` |
| GET | `/health` | 健康检查 + 会话统计 |
| DELETE | `/session/{session_id}` | 清除会话历史 |
| GET | `/session/{session_id}/history` | 获取会话历史 |

## 六、运行方式

### 1. 安装依赖

```bash
cd week04-homework/smart_customer_service
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.deepseek.com/v1
```

### 3. 启动服务

```bash
python main.py
```

服务启动在 `http://0.0.0.0:8000`。

### 4. 运行测试

```bash
# 另开终端
python test_client.py
```

### 测试场景覆盖

- ✅ 基础对话（"你是谁？"）
- ✅ 相对时间理解（"我昨天下的单"）
- ✅ 查订单流程（追问订单号 → 提供订单号 → 查询结果）
- ✅ 直接带订单号查询
- ✅ 查询不存在的订单
- ✅ 申请退款
- ✅ 重复退款校验
