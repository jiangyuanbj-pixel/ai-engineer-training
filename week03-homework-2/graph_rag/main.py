import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from neo4j import GraphDatabase

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==================== 配置 ====================

TEXT = """
A公司是一家大型集团公司。
A公司控股B公司，持股比例为60%。
A公司还控股D公司，持股比例为55%。
B公司控股C公司，持股比例为70%。
B公司控股E公司，持股比例为80%。
C公司控股F公司，持股比例为65%。
D公司控股G公司，持股比例为25%。
M公司控股G公司，持股比例为35%。
"""

"""
A -> (60%)B -> (70%)C -> (65%)F
A -> (60%)B -> (80%)E
A -> (55%)D -> (75)G
"""

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")


# ==================== 数据类 ====================

@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    share: str = ""


@dataclass
class ReasoningPath:
    """可解释性推理路径"""
    question: str
    cypher_query: str = ""
    graph_paths: List[List[str]] = field(default_factory=list)
    rag_context: str = ""
    answer: str = ""


# ==================== GraphRAG 核心 ====================

class GraphRAG:
    """GraphRAG 核心 — 融合文档检索(RAG)与图谱推理(KG)的多跳问答系统"""

    def __init__(self, driver, llm):
        self.driver = driver
        self.llm = llm
        self.index: Optional[VectorStoreIndex] = None

    # -------------------- 图谱构建 --------------------

    async def extract_entities(self, text: str) -> List[Entity]:
        """步骤1: 从文本中提取公司实体"""
        prompt = f"""
从文本中提取公司实体：

文本：{text}

返回JSON格式：
{{
    "entities": [
        {{"name": "公司名", "type": "Company"}}
    ]
}}
只返回JSON，不要其他内容。"""
        response = await self.llm.acomplete(prompt)
        result = self._parse_json(response.text)
        entities = [Entity(e["name"], e["type"]) for e in result.get("entities", [])]
        logger.info(f"  提取到 {len(entities)} 个公司实体")
        return entities

    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """步骤2: 从文本中提取控股关系"""
        entity_names = [e.name for e in entities]
        prompt = f"""
从文本中提取公司间的控股关系：

文本：{text}
公司：{entity_names}

返回JSON格式：
{{
    "relationships": [
        {{"source": "母公司", "target": "子公司", "type": "CONTROLS", "share": "持股比例"}}
    ]
}}
只返回JSON，不要其他内容。"""
        response = await self.llm.acomplete(prompt)
        result = self._parse_json(response.text)
        relationships = []
        for r in result.get("relationships", []):
            if r["source"] in entity_names and r["target"] in entity_names:
                relationships.append(Relationship(r["source"], r["target"], r["type"], r.get("share", "")))
        logger.info(f"  提取到 {len(relationships)} 个控股关系")
        return relationships

    async def build_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """步骤3: 构建公司控股图谱（Neo4j）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for entity in entities:
                session.run(f"MERGE (n:{entity.type} {{name: $name}})", name=entity.name)
            for rel in relationships:
                query = """
                MATCH (a {name: $source})
                MATCH (b {name: $target})
                MERGE (a)-[:CONTROLS {share: $share}]->(b)
                """
                session.run(query, source=rel.source, target=rel.target, share=rel.share)
        logger.info("  公司控股图谱构建完成")

    # -------------------- 图谱查询（多跳推理核心） --------------------

    def find_subsidiaries_with_path(self, parent_company: str) -> List[Dict]:
        """使用图遍历算法查找所有子公司及控股路径"""
        with self.driver.session() as session:
            query = """
            MATCH path = (parent:Company {name: $parent_name})-[:CONTROLS*1..]->(subsidiary:Company)
            RETURN subsidiary.name as subsidiary,
                   length(path) as depth,
                   [node in nodes(path) | node.name] as path_nodes,
                   [rel in relationships(path) | rel.share] as path_shares
            ORDER BY depth, subsidiary.name
            """
            result = session.run(query, parent_name=parent_company)
            return [dict(record) for record in result]

    def find_largest_shareholder(self, company: str) -> Optional[Dict]:
        """查找公司的最大直接股东"""
        with self.driver.session() as session:
            query = """
            MATCH (shareholder:Company)-[r:CONTROLS]->(target:Company {name: $company_name})
            RETURN shareholder.name as shareholder, r.share as share
            ORDER BY r.share DESC
            LIMIT 1
            """
            result = session.run(query, company_name=company)
            record = result.single()
            return dict(record) if record else None

    def find_shareholders_with_path(self, company: str) -> List[Dict]:
        """查找公司的所有上游股东及路径"""
        with self.driver.session() as session:
            query = """
            MATCH path = (shareholder:Company)-[:CONTROLS*1..]->(target:Company {name: $company_name})
            RETURN shareholder.name as shareholder,
                   length(path) as depth,
                   [node in nodes(path) | node.name] as path_nodes,
                   [rel in relationships(path) | rel.share] as path_shares
            ORDER BY depth, shareholder.name
            """
            result = session.run(query, company_name=company)
            return [dict(record) for record in result]

    # -------------------- RAG 文档检索 --------------------

    def build_rag_index(self, text: str):
        """步骤4: 构建 LlamaIndex 向量索引"""
        doc = Document(text=text, metadata={"source": "company_equity"})
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents([doc])
        self.index = VectorStoreIndex(nodes, show_progress=False)
        logger.info("  RAG 文档索引构建完成")

    def rag_query(self, question: str, top_k: int = 3) -> str:
        """RAG 检索相关文档片段"""
        if not self.index:
            return ""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        return "\n".join([n.text for n in nodes])

    # -------------------- 可视化 --------------------

    def visualize_control_structure(self, parent_company: str):
        """可视化控股结构"""
        subsidiaries = self.find_subsidiaries_with_path(parent_company)
        print(f"\n  {parent_company} 的控股结构:")
        print("  " + "=" * 50)
        if not subsidiaries:
            print(f"    {parent_company} 没有子公司")
            return
        levels: Dict[int, list] = {}
        for sub in subsidiaries:
            levels.setdefault(sub["depth"], []).append(sub)
        for depth in sorted(levels.keys()):
            print(f"  第{depth}层子公司:")
            for sub in levels[depth]:
                path_str = " → ".join(sub["path_nodes"])
                print(f"    • {sub['subsidiary']}")
                print(f"      路径: {path_str}")

    # -------------------- 多跳问答（RAG + KG 融合） --------------------

    async def query(self, question: str) -> ReasoningPath:
        """融合 RAG + KG 的多跳问答"""
        path = ReasoningPath(question=question)

        # 1. 实体识别
        company_name = self._extract_company_name(question)

        # 2. RAG 检索
        rag_context = self.rag_query(question)
        path.rag_context = rag_context

        # 3. KG 推理
        graph_context = ""

        if company_name:
            if "子公司" in question or "控股" in question or "旗下" in question:
                # 查找所有子公司
                subsidiaries = self.find_subsidiaries_with_path(company_name)
                path.cypher_query = (
                    f"MATCH path=(parent:Company{{name:'{company_name}'}})"
                    f"-[:CONTROLS*1..]->(s:Company) RETURN *"
                )
                path.graph_paths = [r["path_nodes"] for r in subsidiaries]
                if subsidiaries:
                    levels: Dict[int, list] = {}
                    for sub in subsidiaries:
                        levels.setdefault(sub["depth"], []).append(sub)
                    lines = [f"{company_name}的控股结构:"]
                    for depth in sorted(levels.keys()):
                        lines.append(f"  第{depth}层子公司:")
                        for sub in levels[depth]:
                            p = " → ".join(sub["path_nodes"])
                            #s = " → ".join(sub["path_shares"])
                            s = ",".join([str(n) for n in sub["path_shares"]]) 
                            lines.append(f"    • {sub['subsidiary']} (路径: {p}, 持股: {s})")
                    graph_context = "\n".join(lines)
                else:
                    graph_context = f"{company_name}没有子公司"

            elif "最大股东" in question or "谁控股" in question or "股东" in question:
                largest = self.find_largest_shareholder(company_name)
                path.cypher_query = (
                    f"MATCH (s:Company)-[r:CONTROLS]->(t:Company{{name:'{company_name}'}})"
                    f" RETURN s.name, r.share ORDER BY r.share DESC LIMIT 1"
                )
                if largest:
                    graph_context = f"{company_name}的最大直接股东是{largest['shareholder']}，持股{largest['share']}"
                    path.graph_paths = [[largest["shareholder"], company_name]]
                else:
                    all_sh = self.find_shareholders_with_path(company_name)
                    if all_sh:
                        shortest = min(all_sh, key=lambda x: x["depth"])
                        graph_context = (
                            f"{company_name}的间接股东中，{shortest['shareholder']}通过路径"
                            f" {' → '.join(shortest['path_nodes'])} 控股"
                        )
                        path.graph_paths = [s["path_nodes"] for s in all_sh]
                    else:
                        graph_context = f"未找到{company_name}的股东信息"

            elif "层级" in question or "多少层" in question:
                subsidiaries = self.find_subsidiaries_with_path(company_name)
                path.cypher_query = (
                    f"MATCH path=(parent:Company{{name:'{company_name}'}})"
                    f"-[:CONTROLS*1..]->(s:Company) RETURN *"
                )
                path.graph_paths = [r["path_nodes"] for r in subsidiaries]
                if subsidiaries:
                    max_depth = max(r["depth"] for r in subsidiaries)
                    graph_context = f"{company_name}共有{max_depth}层子公司，总计{len(subsidiaries)}个子公司"
                else:
                    graph_context = f"{company_name}没有子公司"

            else:
                # 通用查询
                subsidiaries = self.find_subsidiaries_with_path(company_name)
                largest = self.find_largest_shareholder(company_name)
                path.cypher_query = (
                    f"MATCH (n)-[r:CONTROLS]-(m) "
                    f"WHERE n.name='{company_name}' OR m.name='{company_name}' RETURN *"
                )
                parts = []
                if subsidiaries:
                    parts.append(f"{company_name}控股的子公司: {', '.join(s['subsidiary'] for s in subsidiaries)}")
                if largest:
                    parts.append(f"{company_name}的最大股东: {largest['shareholder']}({largest['share']})")
                graph_context = "; ".join(parts) if parts else f"未找到关于{company_name}的关系信息"
                path.graph_paths = [r["path_nodes"] for r in subsidiaries]

        # 4. LLM 综合回答
        prompt = f"""基于以下信息回答问题，请给出简洁明确的回答。

问题：{question}

文档检索结果：
{rag_context}

图谱推理结果：
{graph_context}

请综合以上信息回答："""
        response = await self.llm.acomplete(prompt)
        path.answer = response.text.strip()
        return path

    # -------------------- 辅助 --------------------

    @staticmethod
    def _extract_company_name(question: str) -> Optional[str]:
        match = re.search(r"([A-Z]公司)", question)
        return match.group(1) if match else None

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text.strip())


# ==================== 主流程 ====================

async def main():
    """Graph RAG 演示：融合文档检索与图谱推理的多跳问答系统"""
    print("=" * 60)
    print("  Graph RAG: 融合文档检索与图谱推理的多跳问答系统")
    print("=" * 60)

    # 初始化 LLM
    llm = OpenAILike(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        is_chat_model=True,
    )
    Settings.llm = llm
    Settings.embed_model = DashScopeEmbedding()

    # 连接 Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph_rag = GraphRAG(driver, llm)

    try:
        print(f"\n输入的公司控股信息:\n{TEXT.strip()}\n")

        # 步骤1: 提取公司实体
        print("步骤1: 从文本中提取公司实体")
        entities = await graph_rag.extract_entities(TEXT)
        for e in entities:
            print(f"   • {e.name} ({e.type})")

        # 步骤2: 提取控股关系
        print("\n步骤2: 从文本中提取控股关系")
        relationships = await graph_rag.extract_relationships(TEXT, entities)
        for r in relationships:
            share_info = f" ({r.share})" if r.share else ""
            print(f"   • {r.source} --[{r.type}{share_info}]--> {r.target}")

        # 步骤3: 构建控股图谱
        print("\n步骤3: 构建公司控股图谱 (Neo4j)")
        await graph_rag.build_graph(entities, relationships)

        # 步骤4: 可视化控股结构
        print("\n步骤4: 可视化控股结构")
        graph_rag.visualize_control_structure("A公司")

        # 步骤5: 构建 RAG 索引
        print("\n步骤5: 构建文档检索索引 (LlamaIndex)")
        graph_rag.build_rag_index(TEXT)

        # 步骤6: 多跳推理问答
        print("\n步骤6: 多跳推理智能问答 (RAG + KG)")
        print("-" * 60)

        questions = [
            "A公司的子公司有哪些？",
            "A公司的最大股东是谁？",
            "B公司的子公司有哪些？",
            "A公司有多少层级的子公司？",
            "G公司的最大股东是谁？",
            "G公司的的股东有谁？",
        ]

        for question in questions:
            print(f"\n  问: {question}")
            result = await graph_rag.query(question)
            print(f"  答: {result.answer}")
            if result.graph_paths:
                for gp in result.graph_paths:
                    print(f"  推理路径: {' → '.join(gp)}")
            if result.cypher_query:
                print(f"  Cypher: {result.cypher_query[:80]}...")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()
        print("\n演示完成")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
