from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import Dict, List, TypedDict
import os
from openai import OpenAI


print(f"\n🔍 构建向量数据库...")

embeddings_model = 'Embedding-2'
embeddings_url = os.getenv('GLM_API_BASE_URL')
api_key = os.getenv('GLM_API_KEY')
embeddings_client = OpenAI(
    base_url=embeddings_url,
    api_key=api_key
)


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model: str = embeddings_model):
        self.model = model
        self.client = embeddings_client
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding

embeddings = OpenAIEmbeddings(model=embeddings_model)

summaries = [
    '老人与海》以老渔夫远海捕鱼为叙事空间，通过简洁语言和象征意义，赞颂人类精神力量，叩问存在本质，映照现代精神困境',
    '老渔夫圣地亚哥虽物质匮乏，却展现了人类精神的尊严与坚韧',
]
vectorstore = FAISS.from_texts(summaries, embedding=embeddings)
print(vectorstore)


