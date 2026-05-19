# Graph RAG 实验报告：融合文档检索与图谱推理的多跳问答系统

## 1. 实验目标

构建一个融合文档检索（RAG）与知识图谱推理（KG）的多跳问答系统，能够回答需要跨多个数据源推理的复杂问题，如"A公司的最大股东是谁？"。

## 2. 系统架构

```
公司控股关系文本
    ↓
┌──────────────────────────────────────────┐
│         LLM 实体/关系抽取                  │
│  文本 → 提取实体(公司)                    │
│       → 提取关系(CONTROLS + 持股比例)      │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│         Neo4j 图谱构建                     │
│  实体 → 节点（Company）                   │
│  关系 → 边（CONTROLS {share: "60%"}）    │
└──────────────┬───────────────────────────┘
               ↓
         用户问题
            ↓
┌─────────────────┬──────────────────┐
│   RAG 文档检索   │   KG 图谱推理    │
│  (LlamaIndex)   │    (Neo4j)      │
│ 检索相关文本片段  │  Cypher多跳查询  │
└────────┬────────┴────────┬─────────┘
         ↓                 ↓
      [LLM 联合推理（综合 RAG + KG 结果）]
                 ↓
         [可解释性输出]
      - 推理路径展示
      - Cypher 查询语句
      - 图谱关系链
```

## 3. 核心模块

### 3.1 GraphRAG 类

参考 `graphrag_demo.py` 的设计，核心类 `GraphRAG` 包含以下方法：

| 方法 | 说明 |
|------|------|
| `extract_entities(text)` | 通过 LLM 从文本提取公司实体，返回 `Entity` 列表 |
| `extract_relationships(text, entities)` | 通过 LLM 从文本提取控股关系，返回 `Relationship` 列表 |
| `build_graph(entities, relationships)` | 将实体和关系写入 Neo4j，构建图谱 |
| `find_subsidiaries_with_path(company)` | Cypher 多跳遍历，查找所有子公司及控股路径 |
| `find_largest_shareholder(company)` | 查找公司的最大直接股东 |
| `find_shareholders_with_path(company)` | 查找公司的所有上游股东及路径 |
| `build_rag_index(text)` | 使用 LlamaIndex 构建文档向量索引 |
| `rag_query(question)` | RAG 检索相关文档片段 |
| `query(question)` | 融合 RAG + KG 的多跳问答，返回 `ReasoningPath` |
| `visualize_control_structure(company)` | 可视化控股结构 |

### 3.2 LLM 实体/关系抽取

图谱关系**不从文件预定义**，而是通过 LLM 从文本中自动抽取：

- **实体抽取**：从文本中识别公司实体（Company 类型）
- **关系抽取**：从文本中识别公司间的控股关系（CONTROLS），包含持股比例
- **Prompt 设计**：要求 LLM 返回严格 JSON 格式
- **JSON 解析**：兼容 markdown 代码块包裹的 JSON 输出

### 3.3 LlamaIndex 文档检索

- **文档加载**：`Document(text=text)` 直接使用文本构建
- **文档切分**：`SentenceSplitter(chunk_size=256, chunk_overlap=50)`
- **向量索引**：`VectorStoreIndex` + DashScope Embedding
- **检索**：`similarity_top_k=3`

### 3.4 多跳问答流程

1. **实体识别**：从问题中正则匹配公司名（如"A公司"）
2. **RAG 检索**：LlamaIndex 检索相关文档片段
3. **KG 推理**：根据问题类型选择 Cypher 查询
   - "子公司" → `find_subsidiaries_with_path()`
   - "最大股东" → `find_largest_shareholder()`
   - "层级" → `find_subsidiaries_with_path()` 计算深度
4. **LLM 综合**：合并 RAG + KG 结果，生成最终回答
5. **可解释性**：`ReasoningPath` 记录完整推理链

## 4. 多跳推理示例

**问题**：A公司的子公司有哪些？

**推理链**：
1. 实体识别 → "A公司"
2. RAG 检索 → 获取 A公司控股相关文本
3. KG Cypher 查询：`MATCH path=(parent:Company{name:'A公司'})-[:CONTROLS*1..]->(s:Company) RETURN *`
4. 图谱路径：
   - 第1层: B公司 (路径: A公司 → B公司, 持股: 60%)
   - 第1层: D公司 (路径: A公司 → D公司, 持股: 55%)
   - 第2层: C公司 (路径: A公司 → B公司 → C公司)
   - 第2层: E公司 (路径: A公司 → B公司 → E公司)
   - 第2层: G公司 (路径: A公司 → D公司 → G公司)
   - 第3层: F公司 (路径: A公司 → B公司 → C公司 → F公司)
5. LLM 综合生成回答

## 5. 技术难点与解决方案

| 难点 | 解决方案 |
|------|---------|
| LLM 抽取格式不稳定 | 正则提取 JSON + markdown 代码块兼容 |
| RAG 与图谱推理融合 | 两路并行检索 + LLM 综合提示词 |
| 多跳推理 | Cypher `CONTROLS*1..` 路径查询，支持任意跳数 |
| 可解释性 | `ReasoningPath` 数据类记录 Cypher、图谱路径、RAG 上下文 |
| 错误传播 | LLM 交叉验证 RAG 与 KG 结果一致性 |

## 6. 运行方式

### 前置条件
- 启动 Neo4j 服务（默认 `bolt://localhost:7687`）
- 设置 `DASHSCOPE_API_KEY` 环境变量

### 运行
```bash
cd week03-homework-2
pip install -e .
python -m graph_rag.main
```

## 7. 改进方向

1. **实体消歧**：同名实体合并，不同指代归一化
2. **抽取质量校验**：用 LLM 对抽取结果做交叉验证
3. **增量更新**：文本变更时只更新增量部分
4. **更智能的问题路由**：用 LLM 自动判断问题类型，选择合适的 Cypher 查询
5. **联合评分机制**：RAG 相似度 + KG 路径置信度的加权融合
