import os
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.readers.wikipedia import WikipediaReader

# 增加调试日志
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("llama_index").addHandler(logging.StreamHandler(stream=sys.stdout))

#加载wiki
#pip install llama-index-readers-wikipedia
#pip install wikipedia


def main():
    # 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
    # 在根目录可以通过python -m chunking_research.main 运行

    reader = WikipediaReader()
    # Load data from Wikipedia
    document = reader.load_data(pages=["联合国气候变化框架公约",])
    exit()


    Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("QWEN_API_KEY"),
    is_chat_model=True
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )

    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    query_engine = index.as_query_engine()
    response = query_engine.query("量子密码学在网络安全有哪些应用和挑战？")
    print(response)

if __name__ == "__main__":
    main()