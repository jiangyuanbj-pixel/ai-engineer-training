import os
import re
import time
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
    MarkdownNodeParser,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

"""
探索 LlamaIndex 中的句子切片检索（Sentence Window Retrieval）及其参数影响分析
"""


def setup_llm():
    """初始化 LLM 和 Embedding 模型"""
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True,
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192,
    )


def chinese_sentence_split(text: str) -> list:
    """按中文句末标点切分，忽略英文句号"""
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s for s in sentences if s.strip()]


def evaluate_splitter(splitter, documents, question, ground_truth, splitter_name):
    """
    对指定切片方式进行评估
    返回: dict 包含查询结果、检索上下文、评分等
    """
    print(f"\n{'=' * 60}")
    print(f"  📋 切片方式: {splitter_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # 解析文档为节点
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  切片数量: {len(nodes)}")

    if nodes:
        lengths = [len(n.get_content()) for n in nodes]
        print(f"  切片长度: 最短={min(lengths)}, 最长={max(lengths)}, 平均={sum(lengths)/len(lengths):.0f} 字符")

    # 构建索引
    index = VectorStoreIndex(nodes)

    # 根据切片类型选择查询引擎
    if isinstance(splitter, SentenceWindowNodeParser):
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)

    # 查询
    response = query_engine.query(question)
    elapsed = time.time() - start_time

    # 获取检索到的上下文
    source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
    contexts = [sn.node.get_content() for sn in source_nodes] if source_nodes else []

    # 打印结果
    print(f"\n  ❓ 问题: {question}")
    print(f"\n  💡 回答:\n  {str(response)}")
    print(f"\n  📄 参考标准: {ground_truth}")
    print(f"\n  🔍 检索到的上下文片段 ({len(contexts)} 个):")
    for i, ctx in enumerate(contexts, 1):
        print(f"    [{i}] {ctx[:200]}...")

    print(f"\n  ⏱️ 耗时: {elapsed:.2f}s")

    # 上下文冗余度评估（基于上下文总长度）
    total_context_len = sum(len(ctx) for ctx in contexts)
    print(f"  📊 检索上下文总长度: {total_context_len} 字符")

    return {
        "splitter_name": splitter_name,
        "num_nodes": len(nodes),
        "response": str(response),
        "contexts": contexts,
        "elapsed": elapsed,
        "total_context_len": total_context_len,
        "num_source_nodes": len(contexts),
    }


def score_response(result, ground_truth, question):
    """
    对回答质量进行评分（基于 LLM 评判）
    返回各维度评分（1-5分）
    """
    prompt = f"""请评估以下RAG系统的回答质量，从五个维度打分（1-5分）：

问题: {question}
参考标准答案: {ground_truth}
系统回答: {result['response']}
检索上下文总长度: {result['total_context_len']} 字符

请从以下维度评分：
1. 准确性(accuracy)：回答内容是否准确
2. 完整性(completeness)：是否覆盖了参考标准答案的关键点
3. 相关性(relevance)：回答是否切题
4. 冗余度(redundancy)：上下文是否有过多冗余（1=非常精简，5=大量冗余）
5. 上下文连贯性(coherence)：回答是否语义连贯，信息有无断裂或重复矛盾（1=严重断裂，5=非常连贯）

请严格按照以下格式输出（每行一个分值，不要其他内容）：
accuracy=X
completeness=X
relevance=X
redundancy=X
coherence=X"""

    llm_response = Settings.llm.complete(prompt)
    text = llm_response.text.strip()

    scores = {}
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip().lower()
            try:
                scores[key] = int(val.strip())
            except ValueError:
                pass

    return scores


def main():
    """主实验流程"""
    print("🚀 LlamaIndex 切片检索实验")
    print("=" * 60)

    # 初始化 LLM
    setup_llm()
    print("✅ LLM 和 Embedding 模型初始化完成")

    # 加载文档
    docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    documents = SimpleDirectoryReader(docs_path).load_data()
    print(f"✅ 加载了 {len(documents)} 篇文档:")
    for doc in documents:
        print(f"   - {doc.metadata.get('file_name', 'unknown')} ({len(doc.text)} 字符)")

    # 实验问题与参考答案（基于气候变化文档）
    experiments = [
        {
            "question": "气候变化对海平面有什么影响？主要原因是什么？",
            "ground_truth": "全球海平面在1901-2018年间上升了约20厘米，当前速率约3.6mm/年，是20世纪初的2倍以上。主要原因是海水热膨胀和陆地冰盖融化。到2100年可能上升超过1米，威胁全球数亿沿海居民，小岛屿国家面临生存危机。",
        },
        {
            "question": "气候反馈机制有哪些？冰雪-反照率反馈是如何运作的？",
            "ground_truth": "气候系统存在正反馈和负反馈机制。冰雪-反照率反馈是最重要的正反馈之一：温度升高导致冰雪融化，地表反照率降低，吸收更多太阳辐射，进一步加剧升温。云反馈机制更复杂，低云有降温效应，高云有保温效应。",
        },
        {
            "question": "直接空气捕获（DAC）技术的原理是什么？目前面临什么挑战？",
            "ground_truth": "DAC技术能从大气中直接移除CO₂，是最有前景的负排放技术之一。Climeworks在冰岛的Orca工厂每年捕获4000吨CO₂，与玄武岩矿物反应永久封存。当前主要挑战是成本高（每吨600-800美元），预计到2050年可降至100-150美元。",
        },
    ]

    # 定义不同的切片策略
    splitters = {
        # --- SentenceSplitter：按句子边界切分，语义完整性好 ---
        "Sentence(256,overlap=25)": SentenceSplitter(
            chunk_size=256, chunk_overlap=25,
            chunking_tokenizer_fn=chinese_sentence_split,
        ),
        "Sentence(512,overlap=50)": SentenceSplitter(
            chunk_size=512, chunk_overlap=50,
            chunking_tokenizer_fn=chinese_sentence_split,
        ),
        # --- TokenTextSplitter：纯按 token 数切分，不考虑句子边界 ---
        "Token(512,overlap=50)": TokenTextSplitter(
            chunk_size=512, chunk_overlap=50,
        ),
        # --- SentenceWindowNodeParser：以句子为单位切分，检索时用窗口替换上下文 ---
        "SentenceWindow(size=2)": SentenceWindowNodeParser(
            sentence_splitter=chinese_sentence_split,
            window_size=2,
        ),
        "SentenceWindow(size=3)": SentenceWindowNodeParser(
            sentence_splitter=chinese_sentence_split,
            window_size=3,
        ),
    }

    # 运行实验
    all_results = []
    for exp_idx, exp in enumerate(experiments):
        print(f"\n\n{'🎯' * 30}")
        print(f"  实验 {exp_idx + 1}/{len(experiments)}: {exp['question']}")
        print(f"{'🎯' * 30}")

        for splitter_name, splitter in splitters.items():
            try:
                result = evaluate_splitter(
                    splitter, documents, exp["question"], exp["ground_truth"], splitter_name
                )
                result["experiment"] = exp_idx + 1
                result["question"] = exp["question"]
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ {splitter_name} 执行失败: {e}")

    # 评分与汇总
    print(f"\n\n{'📊' * 30}")
    print("  评分汇总")
    print(f"{'📊' * 30}")

    for result in all_results:
        try:
            scores = score_response(result, experiments[result["experiment"] - 1]["ground_truth"], result["question"])
            result["scores"] = scores
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"\n  [{result['splitter_name']}] 实验{result['experiment']}: "
                  f"准确={scores.get('accuracy','N/A')}, "
                  f"完整={scores.get('completeness','N/A')}, "
                  f"相关={scores.get('relevance','N/A')}, "
                  f"冗余={scores.get('redundancy','N/A')}, "
                  f"连贯={scores.get('coherence','N/A')} | 平均={avg:.1f}")
        except Exception as e:
            print(f"\n  [{result['splitter_name']}] 实验{result['experiment']}: 评分失败 ({e})")

    # 打印对比表格
    print(f"\n\n{'=' * 100}")
    print("  📋 完整对比表格")
    print(f"{'=' * 100}")
    print(f"{'切片方式':<28} {'准确':>4} {'完整':>4} {'相关':>4} {'冗余':>4} {'连贯':>4} {'平均':>5} {'节点数':>5} {'上下文长度':>8} {'耗时':>6}")
    print("-" * 100)
    for r in all_results:
        s = r.get("scores", {})
        avg = sum(s.values()) / len(s) if s else 0
        print(f"{r['splitter_name']:<28} "
              f"{s.get('accuracy', '-'):>4} {s.get('completeness', '-'):>4} "
              f"{s.get('relevance', '-'):>4} {s.get('redundancy', '-'):>4} "
              f"{s.get('coherence', '-'):>4} {avg:>5.1f} "
              f"{r['num_nodes']:>5} {r['total_context_len']:>8} {r['elapsed']:>5.1f}s")

    print(f"\n{'=' * 100}")
    print("✅ 实验完成！结果已输出到控制台。")


if __name__ == "__main__":
    main()
