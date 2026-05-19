## 作业一: 探索 LlamaIndex 中的句子切片检索及其参数影响分析
### 作业目标
1. 理解 LlamaIndex 框架中“句子切片”的核心思想与实现机制。
2. 实践使用 LlamaIndex 构建基于句子窗口的检索增强生成（RAG）系统。
3. 对比分析不同参数设置对检索效果和生成质量的影响。
4. 培养对文本分块策略与上下文保留之间权衡的理解。
### 技术背景简介
句子切片是一种特殊的文本分块策略：将文档按句子级别切分，但在检索时返回包含匹配句子的“上下文窗口”（即前后若干句子），从而在精确匹配与上下文完整性之间取得平衡。
### 作业任务与步骤
#### 环境搭建与数据准备
安装必要库：
```
pip install llama-index
pip install llama-index-core
pip install llama-index-llms-openai-like
pip install llama-index-llms-dashscope
pip install llama-index-embeddings-dashscope
```
或者使用uv安装，请在pyproject.toml中添加依赖，然后运行`uv sync`安装。
#### 获取 API KEY
参考链接（https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.27727b085cwI5h&tab=api#/api/?type=model&url=2712195）
获取访问 qwen 模型所需的API KEY ，并保存到 “DASHSCOPE_API_KEY” 环境变量中。
#### 准备一份文本数据集（至少 3 篇长文档，每篇 >1000 字），可选来源：
   - Wikipedia 文章（如“量子计算”、“气候变化”）
   - PDF 学术论文摘要
   - 自定义写作文档（如产品说明书、小说节选）
   - 由大模型生成
要求：文档需包含复杂句式和段落结构，便于观察切片效果。
#### 设置llamaindex 使用 qwen 系列，兼容 OpenAI 接口的大模型和嵌入模型
```python
import os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)
```
#### 使用句子切片
句子切片示例代码
```python
# 句子切片
sentence_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)
evaluate_splitter(sentence_splitter, documents, question, ground_truth, "Sentence")
```
#### 尝试其他切片方式
##### Token 切片
```python
# Token 切片
splitter = TokenTextSplitter(
    chunk_size=32,
    chunk_overlap=4,
    separator="\n"
)
```
##### 句子窗口切片
```python
sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

# 注意：句子窗口切片需要特殊的后处理器
query_engine = index.as_query_engine(
    similarity_top_k=5,
    streaming=True,
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
)

evaluate_splitter(sentence_window_splitter, documents, question, ground_truth, "Sentence Window")
```
##### markdown 切片（选做）（需准备markdown格式的文档，可采用模型生成）
```python
markdown_splitter = MarkdownNodeParser()
evaluate_splitter(markdown_splitter, documents, question, ground_truth, "Markdown")
```
#### 参数对比实验
比较不同参数组合对**检索相关性**和**生成回答质量**的影响
- 检索到的上下文是否包含答案
- LLM 生成的回答是否准确完整
- 上下文冗余程度（主观评分 1–5）
- 制作对比表格或图表展示结果。
#### 报告撰写
请补充[chunking_research/report.md](chunking_research/report.md) 文件，内容包括：
- 哪些参数显著影响效果？为什么？
- chunk_overlap 过大或过小的利弊？
- 如何在“精确检索”与“上下文丰富性”之间权衡
#### 参考资料
- LlamaIndex 官方文档：https://docs.llamaindex.ai
