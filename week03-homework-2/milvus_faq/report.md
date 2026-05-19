# 基于 Milvus 的 FAQ 检索系统 — 实验报告

## 1. 项目概述

基于 **LlamaIndex + Milvus (Lite)** 构建一个 FAQ 语义检索系统。用户输入自然语言问题，系统返回最相关的 FAQ 条目及答案。

## 2. 系统架构

```
FAQ.txt → parse_faq()拆分Q&A → 每个Q&A一份Document → 每Document一个node → DashScope Embedding → Milvus 向量库
                                                                              ↓
用户问题 → DashScope Embedding → Milvus Top-K 检索 → LLM 生成回答 → 返回结果
```

## 3. 文件结构

| 文件 | 职责 |
|------|------|
| `main.py` | 核心逻辑：FAQ 解析、索引构建、查询、热更新检测 |
| `api.py` | FastAPI 接口：RESTful API + 后台热更新守护线程 |
| `FAQ.txt` | FAQ 知识库数据 |

## 4. 核心实现

### 4.1 FAQ 解析与去重
`parse_faq()` 按 `Q:/A:` 格式提取问答对，用 set 去重后构造 `Document`。

### 4.2 语义切分
每份QA生成一个 `Document`，只有一个Node

### 4.3 Milvus 向量存储
使用 **milvus-lite** 本地文件模式，无需部署独立服务。Embedding 维度 1024。

### 4.4 热更新
后台线程每 10 秒检测 FAQ 文件 MD5，发现变更自动重建索引。

## 5. API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/query` | FAQ 检索 |
| POST | `/index/rebuild` | 手动重建索引 |

## 6. 运行方式

```bash
export DASHSCOPE_API_KEY=your_key
cd week03-homework-2

# 命令行模式
python -m milvus_faq.main

# API 模式
uvicorn milvus_faq.api:app --reload
```

## 7. 扩展项

- ✅ 热更新知识库（自动 re-index）
- ✅ RESTful API（FastAPI 封装）
