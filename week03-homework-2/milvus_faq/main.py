"""基于 Milvus 的 FAQ 检索系统 — 核心逻辑"""

import os
import glob
import hashlib
import time
from typing import List, Optional
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore
from pathlib import Path



# ============ 配置 ============
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MILVUS_URI = "./milvus_faq.db"
COLLECTION_NAME = "faq"
# FAQ 文档目录（支持多个 .txt 文件）
FAQ_DIR = "/Users/yuan/workplace/python/ai-engineer-training/week03-homework-2/milvus_faq/faq_doc"
WATCH_INTERVAL = 10  # 热更新检测间隔（秒）


def parse_faq_dir() -> List[Document]:
    """扫描目录下所有 .txt 文件，按 Q&A 对解析为独立 Document"""
    all_documents = []
    seen = set()

    txt_files = sorted(glob.glob(os.path.join(FAQ_DIR, "*.txt")))
    if not txt_files:
        print(f"⚠️ 目录 {FAQ_DIR} 下没有找到 .txt 文件")
        return []

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 Q: 开头切分出每个 Q&A 对
        qa_blocks = []
        current_block = []
        for line in content.split("\n"):
            if line.startswith("Q:") and current_block:
                qa_blocks.append("\n".join(current_block).strip())
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            qa_blocks.append("\n".join(current_block).strip())

        # 每个 Q&A 对生成一个 Document
        for block in qa_blocks:
            block = block.strip()
            if not block or block in seen:
                continue
            seen.add(block)
            all_documents.append(Document(text=block, metadata={"source": os.path.basename(txt_file)}))

    print(f"📄 解析完成: {len(all_documents)} 条唯一 FAQ（来自 {len(txt_files)} 个文件）")
    return all_documents


# ============ 索引引擎 ============
class FAQEngine:
    """FAQ 检索引擎：构建索引、查询、热更新"""

    def __init__(self):
        self.index = None
        self.chat_engine = None
        self.doc_stat = {}  # 带索引文档元数据
        self._last_check_time = 0  # 上次检查时间戳
        self.dir_has_change()  # 初始化文件指纹

    #计算文件md5
    def _file_md5(self, file:str) -> str : 
        with open(file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    #返回表示是否有变更
    def dir_has_change(self)-> bool:
        """
        计算目录下所有文件的md5
        """
        doc_stat = {}
        dir_path = Path(FAQ_DIR)
        for f in dir_path.iterdir():
            if f.is_file():
                stat = f.stat()
                doc_stat[f.name] = self._file_md5(FAQ_DIR + "/" + f.name)
        
        if len(doc_stat) != len(self.doc_stat):
            self.doc_stat = doc_stat
            return True
        
        for f, md5 in doc_stat.items():
            if md5 != self.doc_stat.get(f):
                self.doc_stat = doc_stat
                return True

        return False
                
        

    def _setup_settings(self):
        """配置 LlamaIndex 全局设置"""
        Settings.embed_model = DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
            api_key=API_KEY,
            embed_batch_size=10,
        )
        Settings.llm = OpenAILike(
            model="qwen-plus",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=API_KEY,
            is_chat_model=True,
        )


    def build_index(self):
        """从 milves 目录构建 Milvus 向量索引"""
        self._setup_settings()

        # 1. 扫描目录解析所有 FAQ 文件
        documents = parse_faq_dir()
        if not documents:
            raise ValueError(f"未解析到有效 FAQ 内容，请检查 {FAQ_DIR}")


        # 2. 每个 Document 已是完整 Q&A 对，无需再切分，直接作为节点
        from llama_index.core.schema import TextNode
        nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
        print(f"📋 FAQ 节点: {len(documents)} 个文档 → {len(nodes)} 个节点（每条 Q&A 一个节点）")

        # 3. 创建 Milvus 向量存储并插入数据
        vector_store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=COLLECTION_NAME,
            dim=1024,
            overwrite=True,
            output_fields=["text", "_node_content", "_node_type"],
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )

        # 4. 显式 flush + load（Milvus Lite 写入后需要 flush 才能检索）
        from pymilvus import MilvusClient
        milvus_client = MilvusClient(uri=MILVUS_URI)
        milvus_client.flush(COLLECTION_NAME)
        milvus_client.load_collection(COLLECTION_NAME)

        # 5. 重新从 Milvus 创建索引（确保检索时 collection 已 load）
        vs_loaded = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=COLLECTION_NAME,
            dim=1024,
            overwrite=False,
            output_fields=["text", "_node_content", "_node_type"],
        )
        self.index = VectorStoreIndex.from_vector_store(vs_loaded)

        # FAQ 场景用 query_engine 而非 chat_engine，避免 condense_question 改写查询导致检索偏离
        from llama_index.core import PromptTemplate
        qa_prompt = PromptTemplate(
            "上下文信息如下：\n"
            + "---------------------\n"
            + "{context_str}\n"
            + "---------------------\n"
            + "请严格根据上下文信息回答问题，不要编造内容。如果上下文中没有相关信息，请回答'暂无相关信息'。\n"
            + "问题: {query_str}\n"
            + "回答: "
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            text_qa_template=qa_prompt,
        )
        print(f"✅ 索引构建完成: {len(documents)} 条 FAQ 已写入 Milvus")

    def chat(self, message: str) -> str:
        """检索 FAQ"""
        if not self.query_engine:
            raise RuntimeError("索引未构建，请先调用 build_index()")
        return str(self.query_engine.query(message))

    def reset_chat(self):
        """重置对话历史（当前 query_engine 模式无需操作）"""
        pass

    def check_rebuild(self) -> bool:
        """检查 faq_doc 目录下文件是否变更，有变更则自动重建（按 WATCH_INTERVAL 间隔检测）"""
        now = time.time()
        if now - self._last_check_time < WATCH_INTERVAL:
            return False
        self._last_check_time = now

        if self.dir_has_change():
            print("🔄 检测到 faq_doc 目录文件变更，正在重建索引...")
            self.build_index()
            return True
        return False


# ============ 命令行入口 ============
def main():
    """命令行模式：构建索引并交互式查询"""
    print("=" * 50)
    print("  🔍 Milvus FAQ 检索系统")
    print("=" * 50)

    engine = FAQEngine()
    try:
        engine.build_index()
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        print("💡 请确保已设置 DASHSCOPE_API_KEY")
        return

    # 测试查询
    test_questions = ["如何退货？", "配送要多久？", "支持哪些支付方式？"]
    for q in test_questions:
        print(f"\n❓ {q}")
        print(f"💡 {engine.chat(q)}")

    # 交互模式
    print(f"\n{'=' * 50}")
    print("  💬 交互查询（输入 quit 退出，reset 重置对话）")
    print(f"{'=' * 50}")
    while True:
        try:
            user_input = input("\n🙋 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            engine.reset_chat()
            print("🔄 对话历史已重置")
            continue
        if not user_input:
            continue
        # 热更新：检查文件变更
        engine.check_rebuild()
        try:
            print(f"💡 {engine.chat(user_input)}")
        except Exception as e:
            print(f"❌ 查询失败: {e}")


"""
   为什么不能 python3 main.py
   因为 api.py 里有 from .main import ...（相对导入）。无论 uvicorn 启动命令写在哪个文件里，只要涉及相对导入，就必须用 -m 模块方式运行，否则 Python 不认识 . 前缀的导入。
"""
if __name__ == "__main__":
    #main()
   

    import uvicorn
    uvicorn.run("milvus_faq.api:app", host="0.0.0.0", port=8000, reload=True)
